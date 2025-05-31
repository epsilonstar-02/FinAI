# agents/language_agent/multi_llm_client.py
# Enhanced error details, parameter mapping, and robustness.

import os
import logging
import asyncio
import time
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib

from cachetools import TTLCache
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

# Provider SDKs/libraries
import google.generativeai as genai
import openai
import anthropic
from huggingface_hub import AsyncInferenceClient as HFAsyncInferenceClient # Renamed for clarity
from langchain_community.llms import LlamaCpp # Keep this import style
from transformers import pipeline as hf_transformers_pipeline # For local HF models

from .config import settings, LLMProvider

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Base exception for all LLM client errors."""
    pass

class ProviderError(LLMClientError):
    def __init__(self, provider: str, message: str, original_exception: Optional[Exception] = None):
        self.provider = provider
        self.message = message
        self.original_exception = original_exception
        super().__init__(f"Provider '{provider}' error: {message}" + (f" (Original: {type(original_exception).__name__})" if original_exception else ""))

class AllProvidersFailedError(LLMClientError):
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors # Store full error messages
        error_summary = "; ".join([f"'{p}': {e[:100]}..." if len(e) > 100 else f"'{p}': {e}" for p, e in provider_errors.items()])
        super().__init__(f"All configured providers failed. Error summary: {error_summary}")

# Response cache
_response_cache = TTLCache(maxsize=1000, ttl=settings.CACHE_TTL) if settings.CACHE_RESPONSES else None

def cache_response(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not settings.CACHE_RESPONSES or _response_cache is None:
            return await func(*args, **kwargs)

        # Create a cache key from args and kwargs
        # Exclude 'self' from args if it's a method
        key_args = args[1:] if (args and hasattr(func, '__self__') and args[0] is func.__self__) else args
        
        try:
            # Use a dictionary for key components for consistent ordering via sorted(kwargs.items())
            key_dict = {"args": key_args}
            key_dict.update(kwargs)
            serialized_key_content = json.dumps(key_dict, sort_keys=True, default=str)
            cache_key = hashlib.sha256(serialized_key_content.encode('utf-8')).hexdigest()
        except (TypeError, ValueError) as e:
            logger.warning(f"Cache key generation failed for {func.__name__}: {e}. Skipping cache.")
            return await func(*args, **kwargs)

        if cache_key in _response_cache:
            logger.debug(f"Cache hit for {func.__name__} (key: {cache_key[:10]}...).")
            return _response_cache[cache_key]

        result = await func(*args, **kwargs)
        _response_cache[cache_key] = result
        return result
    return wrapper


class MultiLLMClient:
    def __init__(self):
        self.provider_errors: Dict[str, str] = {}
        # Lazy-loaded local models/pipelines
        self._llama_model: Optional[LlamaCpp] = None
        self._hf_local_pipeline: Optional[Any] = None # Type depends on transformers.pipeline
        
        # Initialize provider SDKs and list available ones
        self.available_providers = self._initialize_providers()
        if not self.available_providers:
            logger.critical("CRITICAL: No LLM providers are available/configured. The Language Agent will not function.")
            # Raise an error or allow app to start in degraded mode, handled by main.py

    def _initialize_providers(self) -> List[LLMProvider]:
        available = []
        provider_init_statuses = {}

        if settings.is_provider_available(LLMProvider.GOOGLE):
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY, transport='rest') # Using REST for wider compatibility
                available.append(LLMProvider.GOOGLE)
                provider_init_statuses[LLMProvider.GOOGLE.value] = "OK"
            except Exception as e:
                provider_init_statuses[LLMProvider.GOOGLE.value] = f"Failed: {e}"
        
        if settings.is_provider_available(LLMProvider.OPENAI):
            try:
                # openai.api_key = settings.OPENAI_API_KEY # Client init handles this
                # if settings.OPENAI_ORGANIZATION:
                #    openai.organization = settings.OPENAI_ORGANIZATION
                available.append(LLMProvider.OPENAI)
                provider_init_statuses[LLMProvider.OPENAI.value] = "OK (Key set)"
            except Exception as e: # Should not happen if key is just set
                provider_init_statuses[LLMProvider.OPENAI.value] = f"Failed: {e}"

        if settings.is_provider_available(LLMProvider.ANTHROPIC):
            available.append(LLMProvider.ANTHROPIC)
            provider_init_statuses[LLMProvider.ANTHROPIC.value] = "OK (Key set)"

        if settings.is_provider_available(LLMProvider.LLAMA):
            available.append(LLMProvider.LLAMA)
            provider_init_statuses[LLMProvider.LLAMA.value] = f"OK (Path: {settings.LLAMA_MODEL_PATH}, lazy load)"
        
        if settings.is_provider_available(LLMProvider.HUGGINGFACE):
            available.append(LLMProvider.HUGGINGFACE)
            status_detail = "local model, lazy load" if settings.HUGGINGFACE_USE_LOCAL else "API key set"
            provider_init_statuses[LLMProvider.HUGGINGFACE.value] = f"OK ({status_detail})"
        
        logger.info(f"LLM Provider Initialization Status: {provider_init_statuses}")
        logger.info(f"Final list of available LLM providers: {[p.value for p in available]}")
        return available

    async def _run_with_retry(self, provider_name_str: str, async_target_callable, *args):
        retryer = AsyncRetrying(
            stop=stop_after_attempt(settings.MAX_RETRIES),
            wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=1, max=10),
            reraise=True, # Reraise the last exception after retries
        )
        try:
            return await retryer.call(async_target_callable, *args)
        except asyncio.TimeoutError as e_timeout:
            msg = f"Request timed out after {settings.TIMEOUT}s (including retries)"
            logger.warning(f"{provider_name_str}: {msg}")
            raise ProviderError(provider_name_str, msg, e_timeout)
        except Exception as e: # Catch-all for other errors after retries
            msg = f"Failed after {settings.MAX_RETRIES} retries: {str(e)}"
            logger.warning(f"{provider_name_str}: {msg} (Type: {type(e).__name__})")
            raise ProviderError(provider_name_str, msg, e)


    async def _execute_google_call(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        config_args = {
            "temperature": gen_args.get("temperature", settings.TEMPERATURE),
            "top_k": gen_args.get("top_k", settings.TOP_K),
            "top_p": gen_args.get("top_p", settings.TOP_P),
            "max_output_tokens": gen_args.get("max_tokens", settings.MAX_TOKENS),
        }
        gemini_config = genai.types.GenerationConfig(**config_args)
        
        response = await asyncio.wait_for(
            model.generate_content_async(prompt, generation_config=gemini_config),
            timeout=settings.TIMEOUT
        )
        if not response.candidates or not response.candidates[0].content.parts:
             raise ProviderError(LLMProvider.GOOGLE.value, "No content in Gemini response.")
        return response.text # .text convenience accessor handles parts

    async def _generate_google(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        return await self._run_with_retry(LLMProvider.GOOGLE.value, self._execute_google_call, prompt, gen_args)


    async def _execute_openai_call(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        # OpenAI Python SDK v1.x uses openai.AsyncOpenAI()
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY, organization=settings.OPENAI_ORGANIZATION)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=gen_args.get("temperature", settings.TEMPERATURE),
                max_tokens=gen_args.get("max_tokens", settings.MAX_TOKENS),
                top_p=gen_args.get("top_p", settings.TOP_P),
            ),
            timeout=settings.TIMEOUT
        )
        if not response.choices or not response.choices[0].message.content:
            raise ProviderError(LLMProvider.OPENAI.value, "No content in OpenAI response.")
        return response.choices[0].message.content

    async def _generate_openai(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        return await self._run_with_retry(LLMProvider.OPENAI.value, self._execute_openai_call, prompt, gen_args)


    async def _execute_anthropic_call(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await asyncio.wait_for(
            client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=gen_args.get("max_tokens", settings.MAX_TOKENS),
                temperature=gen_args.get("temperature", settings.TEMPERATURE),
                top_p=gen_args.get("top_p", settings.TOP_P), # Anthropic supports top_p
                top_k=gen_args.get("top_k", settings.TOP_K), # Anthropic supports top_k
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=settings.TIMEOUT
        )
        if not response.content or not response.content[0].text:
            raise ProviderError(LLMProvider.ANTHROPIC.value, "No content in Anthropic response.")
        return response.content[0].text

    async def _generate_anthropic(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        return await self._run_with_retry(LLMProvider.ANTHROPIC.value, self._execute_anthropic_call, prompt, gen_args)


    def _load_llama_model_sync(self, gen_args: Dict[str, Any]): # Synchronous part
        if self._llama_model is None:
            logger.info(f"Initializing LlamaCpp model from {settings.LLAMA_MODEL_PATH}...")
            try:
                self._llama_model = LlamaCpp(
                    model_path=str(settings.LLAMA_MODEL_PATH), # Ensure path is string
                    n_ctx=settings.LLAMA_N_CTX,
                    n_batch=settings.LLAMA_N_BATCH,
                    temperature=gen_args.get("temperature", settings.TEMPERATURE),
                    top_p=gen_args.get("top_p", settings.TOP_P),
                    # LlamaCpp max_tokens is more like n_predict
                    max_tokens=gen_args.get("max_tokens", settings.MAX_TOKENS), 
                    # top_k=gen_args.get("top_k", settings.TOP_K), # Check LlamaCpp param name for top_k
                    n_gpu_layers=-1 if settings.LLAMA_USE_GPU else 0,
                    verbose=False
                )
                logger.info("LlamaCpp model initialized.")
            except Exception as e:
                self._llama_model = None # Ensure it's None on failure
                logger.error(f"Failed to load LlamaCpp model: {e}", exc_info=True)
                raise ProviderError(LLMProvider.LLAMA.value, f"LlamaCpp model loading failed: {e}", e)
        else: # Update params
            self._llama_model.temperature = gen_args.get("temperature", settings.TEMPERATURE)
            self._llama_model.top_p = gen_args.get("top_p", settings.TOP_P)
            self._llama_model.max_tokens = gen_args.get("max_tokens", settings.MAX_TOKENS)
        return self._llama_model

    async def _execute_llama_call(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        # Load/get model (this might raise ProviderError if loading fails)
        llm_instance = await loop.run_in_executor(None, self._load_llama_model_sync, gen_args)
        if llm_instance is None: # Should have been caught by _load_llama_model_sync
             raise ProviderError(LLMProvider.LLAMA.value, "Llama model instance is None after attempting load.")
        response_text = await loop.run_in_executor(None, llm_instance.invoke, prompt)
        if not response_text:
            raise ProviderError(LLMProvider.LLAMA.value, "Llama model returned empty response.")
        return response_text

    async def _generate_llama(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        return await self._run_with_retry(LLMProvider.LLAMA.value, self._execute_llama_call, prompt, gen_args)


    def _load_hf_pipeline_sync(self, gen_args: Dict[str, Any]): # Synchronous part
        if self._hf_local_pipeline is None:
            logger.info(f"Initializing HuggingFace local pipeline for model {settings.HUGGINGFACE_MODEL}...")
            try:
                # bitsandbytes quantization is complex and environment-dependent.
                # For simplicity, removing direct BitsAndBytesConfig here.
                # Users should configure it via transformers environment or model loading args if needed.
                # device_map="auto" should handle GPU if available.
                self.hf_local_pipeline = hf_transformers_pipeline(
                    "text-generation",
                    model=settings.HUGGINGFACE_MODEL,
                    device_map="auto", 
                    # trust_remote_code=True # May be needed for some models
                )
                logger.info("HuggingFace local pipeline initialized.")
            except Exception as e:
                self._hf_local_pipeline = None
                logger.error(f"Failed to load HuggingFace local pipeline: {e}", exc_info=True)
                raise ProviderError(LLMProvider.HUGGINGFACE.value, f"HF local pipeline loading failed: {e}", e)
        return self._hf_local_pipeline

    async def _execute_hf_local_call(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        text_generator = await loop.run_in_executor(None, self._load_hf_pipeline_sync, gen_args)
        if text_generator is None:
             raise ProviderError(LLMProvider.HUGGINGFACE.value, "HF local pipeline instance is None after attempting load.")

        # Transformers pipeline params
        pipeline_args = {
            "max_new_tokens": gen_args.get("max_tokens", settings.MAX_TOKENS),
            "temperature": gen_args.get("temperature", settings.TEMPERATURE),
            "top_p": gen_args.get("top_p", settings.TOP_P),
            "top_k": gen_args.get("top_k", settings.TOP_K),
            "num_return_sequences": 1,
            "pad_token_id": text_generator.tokenizer.eos_token_id if text_generator.tokenizer.eos_token_id else 50256 # Default for GPT-2
        }
        
        response_list = await loop.run_in_executor(None, lambda: text_generator(prompt, **pipeline_args))
        
        if not response_list or not response_list[0].get("generated_text"):
            raise ProviderError(LLMProvider.HUGGINGFACE.value, "HF local pipeline returned empty response.")
        # The response includes the prompt. Need to extract only generated part if model doesn't do it.
        # Most instruct/chat fine-tuned models output only the completion.
        # If it includes prompt: generated_part = response_list[0]["generated_text"][len(prompt):]
        return response_list[0]["generated_text"]

    async def _execute_hf_api_call(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        client = HFAsyncInferenceClient(token=settings.HUGGINGFACE_API_KEY)
        api_params = {
            "max_new_tokens": gen_args.get("max_tokens", settings.MAX_TOKENS),
            "temperature": gen_args.get("temperature", settings.TEMPERATURE),
            "top_p": gen_args.get("top_p", settings.TOP_P),
            "top_k": gen_args.get("top_k", settings.TOP_K),
            # "return_full_text": False # Often useful to get only generated part
        }
        response_text = await asyncio.wait_for(
            client.text_generation(prompt, model=settings.HUGGINGFACE_MODEL, params=api_params),
            timeout=settings.TIMEOUT
        )
        if not response_text:
            raise ProviderError(LLMProvider.HUGGINGFACE.value, "HF API returned empty response.")
        return response_text

    async def _generate_huggingface(self, prompt: str, gen_args: Dict[str, Any]) -> str:
        provider_name_str = LLMProvider.HUGGINGFACE.value
        if settings.HUGGINGFACE_USE_LOCAL:
            return await self._run_with_retry(f"{provider_name_str}-local", self._execute_hf_local_call, prompt, gen_args)
        else:
            if not settings.HUGGINGFACE_API_KEY:
                raise ProviderError(provider_name_str, "HuggingFace API key not configured for API usage.")
            return await self._run_with_retry(f"{provider_name_str}-api", self._execute_hf_api_call, prompt, gen_args)


    @cache_response
    async def generate_text(
        self, 
        prompt: str, 
        provider_preference: Optional[LLMProvider] = None,
        generation_args: Optional[Dict[str, Any]] = None # Merged, e.g. {"temperature":0.5, "max_tokens":200}
    ) -> Tuple[str, LLMProvider]:
        self.provider_errors.clear()
        effective_gen_args = generation_args or {} # Ensure it's a dict

        providers_to_try: List[LLMProvider] = []
        if provider_preference and provider_preference in self.available_providers:
            providers_to_try.append(provider_preference)
        
        # Add default if available and not already preferred
        if settings.DEFAULT_PROVIDER in self.available_providers and \
           settings.DEFAULT_PROVIDER not in providers_to_try:
            providers_to_try.append(settings.DEFAULT_PROVIDER)
        
        # Add fallbacks if available and not already in list
        for fallback_provider in settings.FALLBACK_PROVIDERS:
            if fallback_provider in self.available_providers and \
               fallback_provider not in providers_to_try:
                providers_to_try.append(fallback_provider)
        
        # Add any other available providers not yet tried (maintains some order)
        for p_avail in self.available_providers:
            if p_avail not in providers_to_try:
                providers_to_try.append(p_avail)

        if not providers_to_try:
            raise LLMClientError("No LLM providers are available or configured to attempt generation.")

        logger.info(f"Attempting generation. Prompt length: {len(prompt)}. Order: {[p.value for p in providers_to_try]}. Args: {effective_gen_args}")

        for current_provider_enum in providers_to_try:
            provider_str = current_provider_enum.value
            try:
                logger.info(f"Attempting generation with: {provider_str}")
                text_result = ""
                if current_provider_enum == LLMProvider.GOOGLE:
                    text_result = await self._generate_google(prompt, effective_gen_args)
                elif current_provider_enum == LLMProvider.OPENAI:
                    text_result = await self._generate_openai(prompt, effective_gen_args)
                elif current_provider_enum == LLMProvider.ANTHROPIC:
                    text_result = await self._generate_anthropic(prompt, effective_gen_args)
                elif current_provider_enum == LLMProvider.LLAMA:
                    text_result = await self._generate_llama(prompt, effective_gen_args)
                elif current_provider_enum == LLMProvider.HUGGINGFACE:
                    text_result = await self._generate_huggingface(prompt, effective_gen_args)
                else:
                    # Should not happen if providers_to_try is built from available_providers
                    logger.warning(f"Unsupported provider enum value encountered: {current_provider_enum}. Skipping.")
                    self.provider_errors[provider_str] = "Unsupported provider type in generation loop."
                    continue 
                
                logger.info(f"Successfully generated text using {provider_str}")
                return text_result.strip(), current_provider_enum
            
            except ProviderError as e:
                # Error already logged by _run_with_retry or specific _generate_X
                self.provider_errors[e.provider] = e.message # e.provider should match provider_str
            except Exception as e_unexpected: # Catch truly unexpected errors from a provider attempt
                logger.error(f"Unexpected error during {provider_str} generation: {e_unexpected}", exc_info=True)
                self.provider_errors[provider_str] = f"Unexpected internal error: {str(e_unexpected)}"
        
        raise AllProvidersFailedError(self.provider_errors)


_multi_llm_client_instance: Optional[MultiLLMClient] = None

def get_multi_llm_client() -> MultiLLMClient:
    global _multi_llm_client_instance
    if _multi_llm_client_instance is None:
        _multi_llm_client_instance = MultiLLMClient()
    return _multi_llm_client_instance

async def generate_text_from_client( # Renamed to avoid conflict with internal client method
    prompt: str, 
    provider: Optional[LLMProvider] = None,
    generation_args: Optional[Dict[str, Any]] = None
) -> Tuple[str, LLMProvider]:
    client = get_multi_llm_client()
    return await client.generate_text(prompt, provider_preference=provider, generation_args=generation_args)