"""Tests for the Orchestrator microservice."""
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi import status
from httpx import AsyncClient, HTTPStatusError, TimeoutException

from orchestrator import client
from orchestrator.models import StepLog, ErrorLog

# Use the test_client fixture from conftest.py

def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "agent": "Orchestrator"}


@pytest.mark.asyncio
async def test_run_text_mode_success(test_client):
    """Test successful orchestration in text mode with all agents responding."""
    # Mock client responses
    with patch.object(client, "call_api", return_value=(150, {"symbol": "AAPL", "price": 150.0})), \
         patch.object(client, "call_scrape", return_value=(200, [{"title": "Test Article"}])), \
         patch.object(client, "call_retrieve", return_value=(180, [{"content": "Apple financial data"}])), \
         patch.object(client, "call_language", return_value=(250, {"text": "Market analysis for AAPL"})):

        response = test_client.post("/run", json={
            "input": "What's the latest on AAPL?", 
            "mode": "text",
            "params": {
                "symbols": "AAPL",
                "topic": "apple stock",
                "query": "apple financial performance"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "Market analysis for AAPL"
        assert len(data["steps"]) == 4  # api, scrape, retrieve, language
        assert len(data["errors"]) == 0
        assert data["audio_url"] is None


@pytest.mark.asyncio
async def test_run_voice_mode_flow(test_client):
    """Test voice mode flow with STT and TTS."""
    # Mock responses
    with patch.object(client, "call_stt") as mock_stt, \
         patch.object(client, "call_api") as mock_api, \
         patch.object(client, "call_language") as mock_lang, \
         patch.object(client, "call_tts") as mock_tts:
        
        # Setup mock return values
        mock_stt.return_value = (100, {"text": "Tell me about AAPL"})
        mock_api.return_value = (150, {"symbol": "AAPL", "price": 150.0})
        mock_lang.return_value = (250, {"text": "Apple stock analysis"})
        mock_tts.return_value = (300, {"audio_url": "https://example.com/audio.mp3"})
        
        # Create a test client that doesn't validate the request body
        response = test_client.post(
            "/run",
            json={
                "input": "",
                "mode": "voice",
                "params": {
                    "audio_bytes": "dGhpcyBpcyBhIHRlc3QgYXVkaW8gZGF0YQ==",  # base64 of "this is a test audio data"
                    "symbols": "AAPL"
                }
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "Apple stock analysis"
        assert data["audio_url"] == "https://example.com/audio.mp3"
        assert len(data["steps"]) == 4  # stt, api, language, tts
        
        # Verify the STT was called with the decoded audio
        assert mock_stt.called
        assert mock_stt.call_args[0][0] == b"this is a test audio data"


@pytest.mark.asyncio
async def test_run_partial_errors(test_client):
    """Test orchestration with some services failing but overall success."""
    api_error = TimeoutException("Connection timeout")
    
    with patch.object(client, "call_api", side_effect=api_error), \
         patch.object(client, "call_scrape", return_value=(200, [{"title": "Test Article"}])), \
         patch.object(client, "call_language", return_value=(250, {"text": "Market analysis"})):

        response = test_client.post("/run", json={
            "input": "What's the latest on AAPL?", 
            "mode": "text",
            "params": {
                "symbols": "AAPL",
                "topic": "apple stock"
            }
        })
        
        assert response.status_code == status.HTTP_206_PARTIAL_CONTENT
        data = response.json()
        assert data["output"] == "Market analysis"
        assert len(data["steps"]) == 2  # scrape, language
        assert len(data["errors"]) == 1  # api error


@pytest.mark.asyncio
async def test_invalid_mode(test_client):
    """Test error when invalid mode is provided."""
    response = test_client.post("/run", json={
        "input": "What's the latest on AAPL?", 
        "mode": "invalid_mode",
        "params": {}
    })
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
