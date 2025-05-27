"""Tests for the Orchestrator Agent."""
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import HTTPStatusError, TimeoutException

from orchestrator.models import RunStep

# Use the test_client fixture from conftest.py

def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "agent": "Orchestrator Agent"}


@pytest.mark.asyncio
@patch("orchestrator.client.api_client.get")
@patch("orchestrator.client.scraping_client.get")
@patch("orchestrator.client.retriever_client.get")
async def test_run_success(mock_retriever_get, mock_scraping_get, mock_api_get, test_client):
    """Test successful orchestration with all agents responding."""
    # Mock responses
    mock_api_get.return_value = {"symbol": "AAPL", "price": 150.0}
    mock_scraping_get.return_value = [
        {"title": "Test Article 1", "url": "https://example.com/1"},
        {"title": "Test Article 2", "url": "https://example.com/2"},
    ]
    mock_retriever_get.return_value = [
        {"content": "Apple financial data", "score": 0.95},
        {"content": "Market analysis", "score": 0.85},
    ]

    response = test_client.post("/run", json={"input": "What's the latest on AAPL?"})
    
    assert response.status_code == 200
    data = response.json()
    assert "output" in data
    assert "steps" in data
    assert len(data["steps"]) == 3


@pytest.mark.asyncio
@patch("orchestrator.client.api_client.get")
@patch("orchestrator.client.scraping_client.get")
@patch("orchestrator.client.retriever_client.get")
async def test_run_partial_failure(mock_retriever_get, mock_scraping_get, mock_api_get, test_client):
    """Test orchestration with one agent failing."""
    # Mock responses
    mock_api_get.side_effect = TimeoutException("Connection timeout")
    mock_scraping_get.return_value = [
        {"title": "Test Article 1", "url": "https://example.com/1"},
    ]
    mock_retriever_get.return_value = [
        {"content": "Market analysis", "score": 0.85},
    ]

    response = test_client.post("/run", json={"input": "What's the latest on AAPL?"})
    
    assert response.status_code == 200
    data = response.json()
    assert "output" in data
    assert "steps" in data
    # Should have 2 steps, not 3 (api_agent failed)
    assert len(data["steps"]) == 2


@pytest.mark.asyncio
@patch("orchestrator.main._call_api_agent")
async def test_api_agent_error_handling(mock_call_api, test_client):
    """Test proper error handling for API agent failures."""
    # Setup
    mock_call_api.side_effect = HTTPStatusError("API error", request=None, response=None)
    
    # Test
    with patch("orchestrator.main._call_scraping_agent") as mock_scraping:
        with patch("orchestrator.main._call_retriever_agent") as mock_retriever:
            mock_scraping.return_value = RunStep(tool="scraping_agent", response={})
            mock_retriever.return_value = RunStep(tool="retriever_agent", response={})
            
            response = test_client.post("/run", json={"input": "AAPL stock price"})
    
    # Verify
    assert response.status_code == 200
    data = response.json()
    assert len(data["steps"]) == 2  # Only scraping and retriever, api failed
