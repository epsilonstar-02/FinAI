"""
Multi-provider LLM client that supports multiple language model providers with fallback mechanisms.
"""
import os
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Callable
import json
from datetime import datetime
import traceback
from functools import wraps
import inspect

# Cache and retry utilities
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

# Import providers
# Google Generative AI
import google.generativeai as genai

# OpenAI
import openai

# Anthropic
import anthropic

# Local models via LangChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# HuggingFace
import huggingface_hub
from huggingface_hub import InferenceClient

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
        super().__init__(f"{provider} error: {message}")


class AllProvidersFailedError(LLMClientError):
    """Exception raised when all providers fail."""
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors
        error_msg = "; ".join([f"{p}: {e}" for p, e in provider_errors.items()])
        super().__init__(f"All providers failed: {error_msg}")


# Response cache
response_cache = TTLCache(maxsize=1000, ttl=settings.CACHE_TTL) if settings.CACHE_RESPONSES else None


def cache_response(func):
    """Decorator to cache responses from LLM providers."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not settings.CACHE_RESPONSES or not response_cache:
            return await func(*args, **kwargs)
        
        # Create cache key from args and kwargs
        # For prompt, only use first 100 chars for the key to avoid very long keys
        cache_items = list(args)
        if len(args) > 0 and isinstance(args[0], str) and len(args[0]) > 100:
            cache_items[0] = args[0][:100]
        
        # Add kwargs to cache items
        for k, v in sorted(kwargs.items()):
            if k == "prompt" and isinstance(v, str) and len(v) > 100:
                cache_items.append((k, v[:100]))
            else:
                cache_items.append((k, v))
        
        # Create a hashable key
        try:
            cache_key = json.dumps(cache_items)
        except (TypeError, ValueError):
            # If we can't create a hashable key, just skip caching
            return await func(*args, **kwargs)
        
        if cache_key in response_cache:
            logger.debug(f"Cache hit for key: {cache_key[:50]}...")
            return response_cache[cache_key]
        
        result = await func(*args, **kwargs)
        try:
            response_cache[cache_key] = result
        except Exception as e:
            logger.warning(f"Failed to cache response: {str(e)}")
        
        return result
    
    return wrapper


class MultiLLMClient:
    """
    Client that can interact with multiple LLM providers with fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the multi-LLM client."""
        self.available_providers = self._initialize_providers()
        self.provider_errors = {}
        
        if not self.available_providers:
            logger.warning("No LLM providers are available. Make sure to set API keys.")
    
    def _initialize_providers(self) -> List[LLMProvider]:
        """Initialize available providers."""
        available = []
        
        # Google Generative AI
        if settings.is_provider_available(LLMProvider.GOOGLE):
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                available.append(LLMProvider.GOOGLE)
            except Exception as e:
                logger.warning(f"Failed to initialize Google Generative AI: {str(e)}")
        
        # OpenAI
        if settings.is_provider_available(LLMProvider.OPENAI):
            try:
                if settings.OPENAI_ORGANIZATION:
                    openai.organization = settings.OPENAI_ORGANIZATION
                openai.api_key = settings.OPENAI_API_KEY
                available.append(LLMProvider.OPENAI)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {str(e)}")
        
        # Anthropic
        if settings.is_provider_available(LLMProvider.ANTHROPIC):
            try:
                # Anthropic client is initialized per-request
                available.append(LLMProvider.ANTHROPIC)
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {str(e)}")
        
        # Llama
        if settings.is_provider_available(LLMProvider.LLAMA):
            try:
                # Llama model is loaded on-demand to save memory
                available.append(LLMProvider.LLAMA)
            except Exception as e:
                logger.warning(f"Failed to initialize Llama: {str(e)}")
        
        # HuggingFace
        if settings.is_provider_available(LLMProvider.HUGGINGFACE):
            try:
                # HuggingFace client is initialized per-request
                available.append(LLMProvider.HUGGINGFACE)
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace: {str(e)}")
        
        return available
    
    async def _generate_google(self, prompt: str) -> str:
        """Generate text using Google Generative AI."""
        try:
            # Initialize the model
            model = genai.GenerativeModel(settings.GEMINI_MODEL)
            
            # Create a chat session
            chat = model.start_chat()
            
            # Send the message and get response with timeout
            response = await asyncio.wait_for(
                chat.send_message_async(
                    prompt,
                    generation_config={
                        "temperature": settings.TEMPERATURE,
                        "top_k": settings.TOP_K,
                        "top_p": settings.TOP_P,
                        "max_output_tokens": settings.MAX_TOKENS,
                    }
                ),
                timeout=settings.TIMEOUT
            )
            
            # Return the text content
            return response.text
        
        except asyncio.TimeoutError:
            raise ProviderError("Google", f"Request timed out after {settings.TIMEOUT} seconds")
        except Exception as e:
            raise ProviderError("Google", str(e))
    
    async def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI."""
        try:
            # Send the message and get response with timeout
            response = await asyncio.wait_for(
                openai.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS,
                    top_p=settings.TOP_P,
                ),
                timeout=settings.TIMEOUT
            )
            
            # Return the text content
            return response.choices[0].message.content
        
        except asyncio.TimeoutError:
            raise ProviderError("OpenAI", f"Request timed out after {settings.TIMEOUT} seconds")
        except Exception as e:
            raise ProviderError("OpenAI", str(e))
    
    async def _generate_anthropic(self, prompt: str) -> str:
        """Generate text using Anthropic."""
        try:
            # Initialize the Anthropic client
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            
            # Send the message and get response with timeout
            response = await asyncio.wait_for(
                client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=settings.MAX_TOKENS,
                    temperature=settings.TEMPERATURE,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ),
                timeout=settings.TIMEOUT
            )
            
            # Return the text content
            return response.content[0].text
        
        except asyncio.TimeoutError:
            raise ProviderError("Anthropic", f"Request timed out after {settings.TIMEOUT} seconds")
        except Exception as e:
            raise ProviderError("Anthropic", str(e))
    
    async def _generate_llama(self, prompt: str) -> str:
        """Generate text using local Llama model."""
        try:
            # Set up the callback manager
            callback_manager = CallbackManager([])
            
            # Initialize the Llama model
            llm = LlamaCpp(
                model_path=settings.LLAMA_MODEL_PATH,
                n_ctx=settings.LLAMA_N_CTX,
                n_batch=settings.LLAMA_N_BATCH,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                max_tokens=settings.MAX_TOKENS,
                callback_manager=callback_manager,
                verbose=False,
                n_gpu_layers=-1 if settings.LLAMA_USE_GPU else 0
            )
            
            # Convert to coroutine for consistent interface
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            
            return response
        
        except Exception as e:
            raise ProviderError("Llama", str(e))
    
    async def _generate_huggingface(self, prompt: str) -> str:
        """Generate text using HuggingFace models."""
        try:
            if settings.HUGGINGFACE_USE_LOCAL:
                # Use local HuggingFace model via Transformers
                from transformers import pipeline
                
                # Create the pipeline
                text_generator = pipeline(
                    "text-generation",
                    model=settings.HUGGINGFACE_MODEL,
                    device_map="auto"
                )
                
                # Convert to coroutine for consistent interface
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: text_generator(
                        prompt,
                        max_length=settings.MAX_TOKENS,
                        temperature=settings.TEMPERATURE,
                        top_p=settings.TOP_P,
                        top_k=settings.TOP_K,
                        num_return_sequences=1
                    )
                )
                
                return response[0]["generated_text"]
            
            else:
                # Use HuggingFace Inference API
                client = InferenceClient(token=settings.HUGGINGFACE_API_KEY)
                
                # Convert to coroutine for consistent interface
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.text_generation(
                        prompt,
                        model=settings.HUGGINGFACE_MODEL,
                        max_length=settings.MAX_TOKENS,
                        temperature=settings.TEMPERATURE,
                        top_p=settings.TOP_P,
                        top_k=settings.TOP_K
                    )
                )
                
                return response
        
        except Exception as e:
            raise ProviderError("HuggingFace", str(e))
    
    @cache_response
    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=settings.RETRY_DELAY, max=10),
        reraise=True
    )
    async def generate_text(self, prompt: str, provider: Optional[LLMProvider] = None) -> str:
        """
        Generate text using the specified provider or the default provider.
        Falls back to other providers if the specified provider fails.
        
        Args:
            prompt: The prompt to send to the model
            provider: Optional specific provider to use
            
        Returns:
            The generated text
            
        Raises:
            LLMClientError: If all providers fail
        """
        # Reset provider errors
        self.provider_errors = {}
        
        # Determine which providers to try
        providers_to_try = []
        
        if provider and provider in self.available_providers:
            # If a specific provider is requested and available, try it first
            providers_to_try.append(provider)
        
        # Then add the default provider if it's not already in the list
        if settings.DEFAULT_PROVIDER in self.available_providers and settings.DEFAULT_PROVIDER not in providers_to_try:
            providers_to_try.append(settings.DEFAULT_PROVIDER)
        
        # Then add fallback providers if they're not already in the list
        for fallback in settings.FALLBACK_PROVIDERS:
            if fallback in self.available_providers and fallback not in providers_to_try:
                providers_to_try.append(fallback)
        
        # Finally, add any other available providers not already in the list
        for available in self.available_providers:
            if available not in providers_to_try:
                providers_to_try.append(available)
        
        # If no providers are available, raise an error
        if not providers_to_try:
            raise LLMClientError("No LLM providers are available. Make sure to set API keys.")
        
        # Try each provider in order
        for provider in providers_to_try:
            try:
                if provider == LLMProvider.GOOGLE:
                    return await self._generate_google(prompt)
                elif provider == LLMProvider.OPENAI:
                    return await self._generate_openai(prompt)
                elif provider == LLMProvider.ANTHROPIC:
                    return await self._generate_anthropic(prompt)
                elif provider == LLMProvider.LLAMA:
                    return await self._generate_llama(prompt)
                elif provider == LLMProvider.HUGGINGFACE:
                    return await self._generate_huggingface(prompt)
                else:
                    # Skip unsupported providers
                    logger.warning(f"Unsupported provider: {provider}")
                    continue
            
            except ProviderError as e:
                # Log the error and continue to the next provider
                logger.warning(f"Provider {provider} failed: {str(e)}")
                self.provider_errors[provider.value] = str(e)
                continue
        
        # If we've tried all providers and all have failed, raise an error
        raise AllProvidersFailedError(self.provider_errors)
    
    def get_provider_errors(self) -> Dict[str, str]:
        """Get the errors that occurred for each provider."""
        return self.provider_errors


# Create a singleton instance
_multi_llm_client = None

def get_multi_llm_client() -> MultiLLMClient:
    """Get the multi-LLM client singleton instance."""
    global _multi_llm_client
    if _multi_llm_client is None:
        _multi_llm_client = MultiLLMClient()
    return _multi_llm_client

async def generate_text(prompt: str, provider: Optional[LLMProvider] = None) -> str:
    """
    Generate text using the multi-LLM client.
    
    Args:
        prompt: The prompt to send to the model
        provider: Optional specific provider to use
        
    Returns:
        The generated text
        
    Raises:
        LLMClientError: If an error occurs in generating text
    """
    client = get_multi_llm_client()
    return await client.generate_text(prompt, provider)
