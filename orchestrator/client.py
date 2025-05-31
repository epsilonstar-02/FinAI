# orchestrator/client.py

"""HTTP client for calling downstream FinAI microservices."""
import time
import logging
import asyncio
from typing import Any, Dict, Optional, Tuple, List, Union, Callable, TypeVar

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception

from orchestrator.config import settings

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

class AgentClient:
    """Client for interacting with various FinAI microservices."""
    
    def __init__(self, default_timeout: int = settings.CLIENT_DEFAULT_TIMEOUT):
        """
        Initialize the agent client.
        
        Args:
            default_timeout: Default timeout in seconds for the HTTP client.
        """
        # The base_url for Orchestrator itself is not used by this client,
        # as this client calls *other* agents.
        self.client = httpx.AsyncClient(timeout=default_timeout)

    @staticmethod
    def _is_retriable_exception(exception: BaseException) -> bool:
        """Determines if an exception is retriable for agent calls."""
        if isinstance(exception, httpx.RequestError): # Covers network errors, timeouts
            logger.warning(f"RequestError encountered: {exception}. Retrying...")
            return True
        if isinstance(exception, httpx.HTTPStatusError):
            # Retry only on 5xx server errors
            if 500 <= exception.response.status_code < 600:
                logger.warning(
                    f"HTTPStatusError {exception.response.status_code} encountered for "
                    f"{exception.request.url}. Retrying..."
                )
                return True
        return False

    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_fixed(settings.RETRY_DELAY),
        retry=retry_if_exception(_is_retriable_exception.__func__) # Use __func__ for staticmethod context with tenacity
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        request_timeout: int, # Specific timeout for this request
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        raise_on_status: bool = True,
    ) -> httpx.Response:
        """
        Makes an HTTP request with retries.
        Returns the raw httpx.Response object on success.
        Raises httpx.RequestError or httpx.HTTPStatusError on failure after retries.
        """
        return await self.client.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            files=files,
            headers=headers,
            timeout=request_timeout,
        )

    async def _execute_agent_call(
        self,
        agent_name: str,
        method: str,
        url: str,
        request_timeout: int,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any]:
        """
        Helper method to execute an agent call, handle exceptions, and format response.
        """
        start_time = time.time()
        try:
            response = await self._make_request(
                method, url, request_timeout, params=params, json_data=json_data, files=files
            )
            response.raise_for_status() # Ensure non-2xx raises HTTPStatusError
            latency_ms = int((time.time() - start_time) * 1000)
            return latency_ms, response.json()
        except httpx.HTTPStatusError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"{agent_name} HTTP error {e.response.status_code} after {latency_ms}ms "
                f"for URL {url}: {e.response.text}"
            )
            if e.response.status_code == 404:
                return latency_ms, {"detail": f"{agent_name} service not found or endpoint missing", "success": False}
            return latency_ms, {"detail": f"{agent_name} service error: {e.response.status_code}", "success": False}
        except httpx.RequestError as e: # Network errors, timeouts after retries
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"{agent_name} network error after {latency_ms}ms for URL {url}: {e}")
            return latency_ms, {"detail": f"{agent_name} service unavailable", "success": False}
        except json.JSONDecodeError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"{agent_name} JSON decode error after {latency_ms}ms for URL {url}: {e}")
            return latency_ms, {"detail": f"Invalid response from {agent_name}", "success": False}
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Unexpected error calling {agent_name} after {latency_ms}ms for URL {url}: {e}",
                exc_info=True
            )
            return latency_ms, {"detail": f"Internal error processing {agent_name} call", "success": False}

    async def call_api_agent(self, symbols: str, timeout: Optional[int] = None) -> Tuple[int, Any]:
        url = f"{settings.API_AGENT_URL}/price"
        effective_timeout = timeout or settings.API_TIMEOUT
        return await self._execute_agent_call(
            "API Agent", "GET", url, effective_timeout, params={"symbols": symbols}
        )

    async def call_scraping_agent(self, topic: str, limit: int = 5, timeout: Optional[int] = None) -> Tuple[int, Any]:
        url = f"{settings.SCRAPING_AGENT_URL}/scrape/news" # Assuming original structure
        effective_timeout = timeout or settings.SCRAPING_TIMEOUT
        return await self._execute_agent_call(
            "Scraping Agent", "GET", url, effective_timeout, params={"topic": topic, "limit": limit}
        )

    async def call_retriever_agent(self, q: str, k: int = 5, timeout: Optional[int] = None) -> Tuple[int, Any]:
        url = f"{settings.RETRIEVER_AGENT_URL}/retrieve"
        effective_timeout = timeout or settings.RETRIEVER_TIMEOUT
        return await self._execute_agent_call(
            "Retriever Agent", "GET", url, effective_timeout, params={"q": q, "k": k}
        )

    async def call_analysis_agent(
        self,
        prices: Dict[str, Any],
        historical: Optional[Dict[str, List[Dict[str, Any]]]] = None, # Made Optional for clarity
        provider: Optional[str] = None,
        include_correlations: bool = False,
        include_risk_metrics: bool = False,
        timeout: Optional[int] = None
    ) -> Tuple[int, Any]:
        url = f"{settings.ANALYSIS_AGENT_URL}/analyze"
        effective_timeout = timeout or settings.ANALYSIS_TIMEOUT
        
        request_data = {
            "prices": prices,
            "include_correlations": include_correlations,
            "include_risk_metrics": include_risk_metrics
        }
        if historical: # Add only if provided
            request_data["historical"] = historical
        if provider:
            request_data["provider"] = provider
            
        return await self._execute_agent_call(
            "Analysis Agent", "POST", url, effective_timeout, json_data=request_data
        )

    async def call_language_agent(
        self,
        query: str,
        context: Dict[str, Any],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> Tuple[int, Any]:
        url = f"{settings.LANGUAGE_AGENT_URL}/generate"
        effective_timeout = timeout or settings.LANGUAGE_TIMEOUT
        
        request_data = {"query": query, "context": context}
        if model: request_data["model"] = model
        if max_tokens is not None: request_data["max_tokens"] = max_tokens
        if temperature is not None: request_data["temperature"] = temperature
            
        return await self._execute_agent_call(
            "Language Agent", "POST", url, effective_timeout, json_data=request_data
        )

    async def call_voice_agent_stt(
        self,
        file_bytes: bytes,
        provider: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Tuple[int, Any]:
        url = f"{settings.VOICE_AGENT_URL}/stt"
        effective_timeout = timeout or settings.VOICE_TIMEOUT
        
        files = {"file": ("audio.wav", file_bytes, "audio/wav")}
        params_query = {} # Query parameters for STT provider
        if provider:
            params_query["provider"] = provider
            
        return await self._execute_agent_call(
            "Voice Agent (STT)", "POST", url, effective_timeout, files=files, params=params_query
        )

    async def call_voice_agent_tts(
        self,
        text: str,
        provider: Optional[str] = None,
        voice: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        pitch: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> Tuple[int, Any]:
        url = f"{settings.VOICE_AGENT_URL}/tts"
        effective_timeout = timeout or settings.VOICE_TIMEOUT
        
        params_query = {"response_format": "base64"} # Query parameters
        request_data = {"text": text} # JSON body
        
        if provider: request_data["provider"] = provider
        if voice: request_data["voice"] = voice
        if speaking_rate is not None: request_data["speaking_rate"] = speaking_rate
        if pitch is not None: request_data["pitch"] = pitch
            
        return await self._execute_agent_call(
            "Voice Agent (TTS)", "POST", url, effective_timeout, params=params_query, json_data=request_data
        )

    async def call_with_fallback(
        self,
        primary_func: Callable[..., Any],
        fallback_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Call a function with fallback if the primary call fails."""
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function {primary_func.__name__} failed: {e}. Trying fallback {fallback_func.__name__}...")
            try:
                return await fallback_func(*args, **kwargs)
            except Exception as fallback_e:
                logger.error(f"Fallback function {fallback_func.__name__} also failed: {fallback_e}")
                raise # Re-raise the fallback error, or could choose to raise original error 'e'

# Global instance
agent_client = AgentClient()

# Aliases for backward compatibility (and potentially shorter names if preferred)
call_api = agent_client.call_api_agent
call_scrape = agent_client.call_scraping_agent
call_retrieve = agent_client.call_retriever_agent
call_analysis = agent_client.call_analysis_agent
call_language = agent_client.call_language_agent
call_stt = agent_client.call_voice_agent_stt
call_tts = agent_client.call_voice_agent_tts
call_with_fallback = agent_client.call_with_fallback # This one was fine