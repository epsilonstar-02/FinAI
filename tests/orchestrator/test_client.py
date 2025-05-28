"""Test client module for the Orchestrator Agent."""
import pytest
import httpx
import respx
from typing import Dict, Any
from unittest.mock import patch
import tenacity
from orchestrator import client
from orchestrator.config import settings


@pytest.mark.asyncio
async def test_call_api_success():
    """Test call_api returns expected response."""
    expected_data = {"symbol": "AAPL", "price": 150.0}
    
    with respx.mock:
        respx.get(f"{settings.API_AGENT_URL}/price").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_api("AAPL")
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data


@pytest.mark.asyncio
async def test_call_api_timeout():
    """Test call_api raises on timeout after retries."""
    with respx.mock:
        # Mock multiple timeout responses to match the retry count
        mock_route = respx.get(f"{settings.API_AGENT_URL}/price")
        mock_route.side_effect = httpx.TimeoutException("Timeout")
        
        # The client should raise RetryError after all retries
        with pytest.raises(tenacity.RetryError) as exc_info:
            await client.call_api("AAPL")
        
        # Verify the underlying cause is a TimeoutException
        assert isinstance(exc_info.value.last_attempt.exception(), httpx.TimeoutException)
        
        # Verify the request was retried the expected number of times (3)
        assert mock_route.called
        assert mock_route.call_count == 3  # 1 initial attempt + 2 retries


@pytest.mark.asyncio
async def test_call_scrape_success():
    """Test call_scrape returns expected response."""
    expected_data = [{"title": "News 1"}, {"title": "News 2"}]
    
    with respx.mock:
        respx.get(f"{settings.SCRAPING_AGENT_URL}/scrape/news").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_scrape("finance", 2)
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data


@pytest.mark.asyncio
async def test_call_retrieve_success():
    """Test call_retrieve returns expected response."""
    expected_data = [{"content": "Info 1"}, {"content": "Info 2"}]
    
    with respx.mock:
        respx.get(f"{settings.RETRIEVER_AGENT_URL}/retrieve").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_retrieve("finance stocks", 2)
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data


@pytest.mark.asyncio
async def test_call_analysis_success():
    """Test call_analysis returns expected response."""
    expected_data = {"trend": "upward", "prediction": 160.0}
    
    with respx.mock:
        respx.post(f"{settings.ANALYSIS_AGENT_URL}/analyze").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_analysis({"AAPL": [150.0, 151.0]})
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data


@pytest.mark.asyncio
async def test_call_language_success():
    """Test call_language returns expected response."""
    expected_data = {"text": "Generated text about finance"}
    
    with respx.mock:
        respx.post(f"{settings.LANGUAGE_AGENT_URL}/generate").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_language("finance", {"data": "context"})
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data


@pytest.mark.asyncio
async def test_call_stt_success():
    """Test call_stt returns expected response."""
    expected_data = {"text": "Convert this audio to text"}
    
    with respx.mock:
        respx.post(f"{settings.VOICE_AGENT_URL}/stt").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_stt(b"audio_bytes")
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data


@pytest.mark.asyncio
async def test_call_tts_success():
    """Test call_tts returns expected response."""
    expected_data = {"audio_url": "https://example.com/audio.mp3"}
    
    with respx.mock:
        respx.post(f"{settings.VOICE_AGENT_URL}/tts").respond(
            status_code=200, json=expected_data
        )
        
        latency_ms, response = await client.call_tts("Convert this text to speech")
        
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0
        assert response == expected_data
