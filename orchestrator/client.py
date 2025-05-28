"""HTTP client for calling downstream services."""
import time
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_fixed

from orchestrator.config import settings


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def _make_request(
    method: str,
    url: str,
    timeout: int,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Any]:
    """Make an HTTP request with retries and return latency and response."""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                files=files,
            )
            response.raise_for_status()
            latency_ms = int((time.time() - start_time) * 1000)
            return latency_ms, response.json()
    except httpx.TimeoutException as e:
        raise e
    except Exception as e:
        raise e


async def call_api(symbols: str, timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call API Agent to get price data for given symbols."""
    url = f"{settings.API_AGENT_URL.rstrip('/')}/price"
    return await _make_request(
        "GET", url, timeout or settings.TIMEOUT, params={"symbol": symbols}
    )


async def call_scrape(topic: str, limit: int, timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call Scraping Agent to get news articles on a topic."""
    url = f"{settings.SCRAPING_AGENT_URL.rstrip('/')}/scrape/news"
    return await _make_request(
        "GET", url, timeout or settings.TIMEOUT, params={"topic": topic, "limit": limit}
    )


async def call_retrieve(q: str, k: int, timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call Retriever Agent to get relevant information from the vector store."""
    url = f"{settings.RETRIEVER_AGENT_URL.rstrip('/')}/retrieve"
    return await _make_request(
        "GET", url, timeout or settings.TIMEOUT, params={"q": q, "k": k}
    )


async def call_analysis(prices: Dict[str, Any], historical: bool = False, timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call Analysis Agent to analyze price data."""
    url = f"{settings.ANALYSIS_AGENT_URL.rstrip('/')}/analyze"
    return await _make_request(
        "POST", url, timeout or settings.TIMEOUT, json_data={"prices": prices, "historical": historical}
    )


async def call_language(query: str, context: Dict[str, Any], timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call Language Agent to generate text based on context."""
    url = f"{settings.LANGUAGE_AGENT_URL.rstrip('/')}/generate"
    return await _make_request(
        "POST", url, timeout or settings.TIMEOUT, json_data={"query": query, "context": context}
    )


async def call_stt(file_bytes: bytes, timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call Voice Agent's speech-to-text endpoint."""
    url = f"{settings.VOICE_AGENT_URL.rstrip('/')}/stt"
    files = {"file": file_bytes}
    return await _make_request(
        "POST", url, timeout or settings.TIMEOUT, files=files
    )


async def call_tts(text: str, timeout: Optional[int] = None) -> Tuple[int, Any]:
    """Call Voice Agent's text-to-speech endpoint."""
    url = f"{settings.VOICE_AGENT_URL.rstrip('/')}/tts"
    return await _make_request(
        "POST", url, timeout or settings.TIMEOUT, json_data={"text": text}
    )