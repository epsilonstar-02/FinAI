"""
Multi-provider LLM client that supports multiple language model providers with fallback mechanisms.
"""
import os
import logging
import asyncio
import time
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
import inspect

from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying, # For async functions
)

# Import providers
import google.generativeai as genai
import openai
import anthropic

# Config and exceptions
from .config import settings, LLMProvider

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for all LLM client errors."""
    pass


class ProviderError(LLMClientError):
    """Exception raised when a specific provider fails."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"Provider '{provider}' error: {message}")


class AllProvidersFailedError(LLMClientError):
    """Exception raised when all providers fail."""
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors
        error_msg = "; ".join([f"'{p}': {e}" for p, e in provider_errors.items()])
        super().__init__(f"All configured providers failed. Errors: {error_msg}")


# Response cache
response_cache = TTLCache(maxsize=1000, ttl=settings.CACHE_TTL) if settings.CACHE_RESPONSES else None


def cache_response(func):
    """Decorator to cache responses from LLM providers."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not settings.CACHE_RESPONSES or not response_cache:
            return await func(*args, **kwargs)

        key_components = []
        # Exclude 'self' if it's a method of a class
        start_index = 0
        if args and hasattr(func, '__self__') and args[0] is func.__self__:
            start_index = 1 
        key_components.extend(list(args[start_index:]))
        
        for k, v in sorted(kwargs.items()):
            key_components.append((k, v))
        
        try:
            serialized_key_content = json.dumps(key_components, sort_keys=True, default=str)
            cache_key = hashlib.sha256(serialized_key_content.encode('utf-8')).hexdigest()
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to create cache key for {func.__name__}: {e}. Skipping cache.")
            return await func(*args, **kwargs)

        if cache_key in response_cache:
            logger.debug(f"Cache hit for key: {cache_key[:10]}...")
            return response_cache[cache_key]

        result = await func(*args, **kwargs)
        response_cache[cache_key] = result # Store the actual result
        return result
    return wrapper


class MultiLLMClient:
    """
    Client that can interact with multiple LLM providers with fallback mechanisms.
    """
    
    def __init__(self):
        self.available_providers = self._initialize_providers()
        self.provider_errors: Dict[str, str] = {}
        
        # For caching initialized local models/pipelines
        self.llama_model = None
        self.hf_local_pipeline = None
        
        if not self.available_providers:
            logger.warning("No LLM providers seem to be fully available/configured based on settings. API key or model path issues might exist.")

    def _initialize_providers(self) -> List[LLMProvider]:
        """Initialize SDKs for available providers and list them."""
        available = []
        
        if settings.is_provider_available(LLMProvider.GOOGLE):
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                available.append(LLMProvider.GOOGLE)
                logger.info("Google Generative AI provider initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Generative AI: {str(e)}")
        
        if settings.is_provider_available(LLMProvider.OPENAI):
            try:
                openai.api_key = settings.OPENAI_API_KEY
                if settings.OPENAI_ORGANIZATION:
                    openai.organization = settings.OPENAI_ORGANIZATION
                available.append(LLMProvider.OPENAI)
                logger.info("OpenAI provider initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {str(e)}")
        
        if settings.is_provider_available(LLMProvider.ANTHROPIC):
            # Anthropic client is initialized per-request with API key
            available.append(LLMProvider.ANTHROPIC)
            logger.info("Anthropic provider configured (client initialized per request).")

        if settings.is_provider_available(LLMProvider.LLAMA):
            available.append(LLMProvider.LLAMA)
            logger.info(f"Llama provider configured (model at {settings.LLAMA_MODEL_PATH} will be loaded on first use).")
        
        if settings.is_provider_available(LLMProvider.HUGGINGFACE):
            available.append(LLMProvider.HUGGINGFACE)
            if settings.HUGGINGFACE_USE_LOCAL:
                logger.info(f"HuggingFace local provider configured (model {settings.HUGGINGFACE_MODEL} will be loaded on first use).")
            else:
                logger.info("HuggingFace API provider configured.")
        
        return available

    async def _run_with_retry(self, provider_name: str, async_target_callable, *args):
        """Helper to run an async callable with retry logic and wrap exceptions."""
        retryer = AsyncRetrying(
            stop=stop_after_attempt(settings.MAX_RETRIES),
            wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=settings.RETRY_DELAY, max=10), # Cap max wait
            reraise=True,
        )
        try:
            return await retryer.call(async_target_callable, *args)
        except asyncio.TimeoutError:
            msg = f"Request timed out after {settings.TIMEOUT}s (including retries)"
            logger.warning(f"{provider_name}: {msg}")
            raise ProviderError(provider_name, msg)
        except Exception as e:
            # Catch-all for other errors after retries
            msg = f"Failed after {settings.MAX_RETRIES} retries: {str(e)}"
            logger.warning(f"{provider_name}: {msg} - {type(e).__name__}")
            # logger.debug(traceback.format_exc()) # For more detailed debugging
            raise ProviderError(provider_name, msg)

    async def _execute_google_call(self, prompt: str, generation_args: Dict[str, Any]):
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        # Gemini uses "max_output_tokens", so map "max_tokens"
        config_args = {
            "temperature": generation_args.get("temperature", settings.TEMPERATURE),
            "top_k": generation_args.get("top_k", settings.TOP_K),
            "top_p": generation_args.get("top_p", settings.TOP_P),
            "max_output_tokens": generation_args.get("max_tokens", settings.MAX_TOKENS),
        }
        # Gemini API expects GenerationConfig object for these
        gemini_config = genai.types.GenerationConfig(**config_args)
        
        response = await asyncio.wait_for(
            model.generate_content_async(prompt, generation_config=gemini_config),
            timeout=settings.TIMEOUT
        )
        return response.text

    async def _generate_google(self, prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> str:
        final_gen_args = generation_args or {}
        return await self._run_with_retry(LLMProvider.GOOGLE.value, self._execute_google_call, prompt, final_gen_args)

    async def _execute_openai_call(self, prompt: str, generation_args: Dict[str, Any]):
        # OpenAI uses "max_tokens" directly
        response = await asyncio.wait_for(
            openai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=generation_args.get("temperature", settings.TEMPERATURE),
                max_tokens=generation_args.get("max_tokens", settings.MAX_TOKENS),
                top_p=generation_args.get("top_p", settings.TOP_P),
                # top_k not directly supported in chat completions, managed by top_p
            ),
            timeout=settings.TIMEOUT
        )
        return response.choices[0].message.content

    async def _generate_openai(self, prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> str:
        final_gen_args = generation_args or {}
        return await self._run_with_retry(LLMProvider.OPENAI.value, self._execute_openai_call, prompt, final_gen_args)

    async def _execute_anthropic_call(self, prompt: str, generation_args: Dict[str, Any]):
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) # Use Async client
        # Anthropic uses "max_tokens" directly
        response = await asyncio.wait_for(
            client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=generation_args.get("max_tokens", settings.MAX_TOKENS),
                temperature=generation_args.get("temperature", settings.TEMPERATURE),
                # top_p and top_k might be supported, check Anthropic docs if needed
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=settings.TIMEOUT
        )
        return response.content[0].text

    async def _generate_anthropic(self, prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> str:
        final_gen_args = generation_args or {}
        return await self._run_with_retry(LLMProvider.ANTHROPIC.value, self._execute_anthropic_call, prompt, final_gen_args)

    def _load_llama_model_sync(self, generation_args: Dict[str, Any]):
        # This part runs in an executor, so synchronous imports are fine here.
        from langchain_community.llms import LlamaCpp # Updated import
        
        if self.llama_model is None:
            logger.info(f"Initializing LlamaCpp model from {settings.LLAMA_MODEL_PATH}...")
            self.llama_model = LlamaCpp(
                model_path=settings.LLAMA_MODEL_PATH,
                n_ctx=settings.LLAMA_N_CTX,
                n_batch=settings.LLAMA_N_BATCH,
                temperature=generation_args.get("temperature", settings.TEMPERATURE),
                top_p=generation_args.get("top_p", settings.TOP_P),
                max_tokens=generation_args.get("max_tokens", settings.MAX_TOKENS),
                # top_k might be specific to LlamaCpp version/params
                n_gpu_layers=-1 if settings.LLAMA_USE_GPU else 0,
                verbose=False # Keep logs cleaner
            )
            logger.info("LlamaCpp model initialized.")
        else: # Update params if model already loaded
            self.llama_model.temperature = generation_args.get("temperature", settings.TEMPERATURE)
            self.llama_model.top_p = generation_args.get("top_p", settings.TOP_P)
            self.llama_model.max_tokens = generation_args.get("max_tokens", settings.MAX_TOKENS)
        return self.llama_model

    async def _execute_llama_call(self, prompt: str, generation_args: Dict[str, Any]):
        loop = asyncio.get_event_loop()
        llm = await loop.run_in_executor(None, self._load_llama_model_sync, generation_args)
        response = await loop.run_in_executor(None, llm.invoke, prompt)
        return response

    async def _generate_llama(self, prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> str:
        final_gen_args = generation_args or {}
        return await self._run_with_retry(LLMProvider.LLAMA.value, self._execute_llama_call, prompt, final_gen_args)

    def _load_hf_pipeline_sync(self, generation_args: Dict[str, Any]):
        # This part runs in an executor.
        from transformers import pipeline
        
        if self.hf_local_pipeline is None:
            logger.info(f"Initializing HuggingFace local pipeline for model {settings.HUGGINGFACE_MODEL}...")
            quantization_config = None
            if settings.HUGGINGFACE_QUANTIZE == "4bit":
                # Requires bitsandbytes
                # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                logger.info("4bit quantization requested for HF local (ensure bitsandbytes is installed and configured).")
            elif settings.HUGGINGFACE_QUANTIZE == "8bit":
                # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                 logger.info("8bit quantization requested for HF local (ensure bitsandbytes is installed and configured).")

            self.hf_local_pipeline = pipeline(
                "text-generation",
                model=settings.HUGGINGFACE_MODEL,
                device_map="auto", # Uses CUDA if available, then MPS, then CPU
                # quantization_config=quantization_config # If using bitsandbytes
            )
            logger.info("HuggingFace local pipeline initialized.")
        return self.hf_local_pipeline

    async def _execute_hf_local_call(self, prompt: str, generation_args: Dict[str, Any]):
        loop = asyncio.get_event_loop()
        text_generator = await loop.run_in_executor(None, self._load_hf_pipeline_sync, generation_args)
        
        # Transformers pipeline uses "max_new_tokens" for controlling output length beyond prompt
        # and "max_length" for total length. "max_tokens" usually means new tokens.
        response = await loop.run_in_executor(
            None,
            lambda: text_generator(
                prompt,
                max_new_tokens=generation_args.get("max_tokens", settings.MAX_TOKENS),
                temperature=generation_args.get("temperature", settings.TEMPERATURE),
                top_p=generation_args.get("top_p", settings.TOP_P),
                top_k=generation_args.get("top_k", settings.TOP_K),
                num_return_sequences=1
            )
        )
        return response[0]["generated_text"]

    async def _execute_hf_api_call(self, prompt: str, generation_args: Dict[str, Any]):
        from huggingface_hub import AsyncInferenceClient # Use async client

        client = AsyncInferenceClient(token=settings.HUGGINGFACE_API_KEY)
        # HF Inference API uses "max_new_tokens"
        response_text = await asyncio.wait_for(
            client.text_generation(
                prompt,
                model=settings.HUGGINGFACE_MODEL,
                max_new_tokens=generation_args.get("max_tokens", settings.MAX_TOKENS),
                temperature=generation_args.get("temperature", settings.TEMPERATURE),
                top_p=generation_args.get("top_p", settings.TOP_P),
                top_k=generation_args.get("top_k", settings.TOP_K)
            ),
            timeout=settings.TIMEOUT
        )
        return response_text

    async def _generate_huggingface(self, prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> str:
        final_gen_args = generation_args or {}
        if settings.HUGGINGFACE_USE_LOCAL:
            return await self._run_with_retry(f"{LLMProvider.HUGGINGFACE.value}-local", self._execute_hf_local_call, prompt, final_gen_args)
        else:
            return await self._run_with_retry(f"{LLMProvider.HUGGINGFACE.value}-api", self._execute_hf_api_call, prompt, final_gen_args)

    @cache_response
    async def generate_text(
        self, 
        prompt: str, 
        provider_preference: Optional[LLMProvider] = None,
        generation_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, LLMProvider]:
        """
        Generate text using the specified provider or the default provider.
        Falls back to other providers if the initial provider(s) fail.
        
        Args:
            prompt: The prompt to send to the model.
            provider_preference: Optional specific provider to try first.
            generation_args: Optional dictionary of generation parameters 
                             (e.g., "temperature", "max_tokens") to override settings.
            
        Returns:
            A tuple containing (generated_text: str, used_provider: LLMProvider).
            
        Raises:
            AllProvidersFailedError: If all configured and attempted providers fail.
            LLMClientError: If no providers are available/configured.
        """
        self.provider_errors.clear() # Clear errors from previous calls
        
        providers_to_try: List[LLMProvider] = []
        
        # Build the order of providers to attempt
        if provider_preference and provider_preference in self.available_providers:
            providers_to_try.append(provider_preference)
        
        if settings.DEFAULT_PROVIDER in self.available_providers and settings.DEFAULT_PROVIDER not in providers_to_try:
            providers_to_try.append(settings.DEFAULT_PROVIDER)
        
        for fallback in settings.FALLBACK_PROVIDERS:
            if fallback in self.available_providers and fallback not in providers_to_try:
                providers_to_try.append(fallback)
        
        # Add any remaining available providers not yet in the list
        for p_avail in self.available_providers:
            if p_avail not in providers_to_try:
                providers_to_try.append(p_avail)

        if not providers_to_try:
            if not self.available_providers:
                 raise LLMClientError("No LLM providers are available or configured correctly. Check API keys and model paths.")
            else:
                 # This case should ideally not be hit if available_providers is populated.
                 raise LLMClientError("Could not determine any provider to try, despite some being available.")

        logger.info(f"Attempting generation with providers in order: {[p.value for p in providers_to_try]}")

        for current_provider in providers_to_try:
            try:
                logger.info(f"Attempting generation with: {current_provider.value}")
                text_result = ""
                if current_provider == LLMProvider.GOOGLE:
                    text_result = await self._generate_google(prompt, generation_args)
                elif current_provider == LLMProvider.OPENAI:
                    text_result = await self._generate_openai(prompt, generation_args)
                elif current_provider == LLMProvider.ANTHROPIC:
                    text_result = await self._generate_anthropic(prompt, generation_args)
                elif current_provider == LLMProvider.LLAMA:
                    text_result = await self._generate_llama(prompt, generation_args)
                elif current_provider == LLMProvider.HUGGINGFACE:
                    text_result = await self._generate_huggingface(prompt, generation_args)
                else:
                    logger.warning(f"Unsupported provider enum value: {current_provider}. Skipping.")
                    self.provider_errors[str(current_provider)] = "Unsupported provider type"
                    continue
                
                logger.info(f"Successfully generated text using {current_provider.value}")
                return text_result, current_provider # Return text and the successful provider
            
            except ProviderError as e:
                logger.warning(str(e)) # Already includes provider name
                self.provider_errors[e.provider] = e.message
            except Exception as e: # Catch unexpected errors from a provider attempt
                logger.error(f"Unexpected error during {current_provider.value} generation: {str(e)}", exc_info=True)
                self.provider_errors[current_provider.value] = f"Unexpected: {str(e)}"
        
        raise AllProvidersFailedError(self.provider_errors)

# Singleton instance management
_multi_llm_client: Optional[MultiLLMClient] = None

def get_multi_llm_client() -> MultiLLMClient:
    global _multi_llm_client
    if _multi_llm_client is None:
        _multi_llm_client = MultiLLMClient()
    return _multi_llm_client

async def generate_text(
    prompt: str, 
    provider: Optional[LLMProvider] = None,
    generation_args: Optional[Dict[str, Any]] = None
) -> Tuple[str, LLMProvider]:
    client = get_multi_llm_client()
    return await client.generate_text(prompt, provider_preference=provider, generation_args=generation_args)