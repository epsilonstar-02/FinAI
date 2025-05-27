import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents.language_agent.main import app
from agents.language_agent.llm_client import LLMClientError


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app
    """
    return TestClient(app)


@pytest.fixture
def mock_generate_text():
    """
    Mock the generate_text function to avoid making actual API calls
    """
    with patch("agents.language_agent.main.generate_text", new_callable=AsyncMock) as mock:
        mock.return_value = "This is a mock response from the language model."
        yield mock


@pytest.fixture
def mock_generate_text_error():
    """
    Mock the generate_text function to simulate an error
    """
    with patch("agents.language_agent.main.generate_text", new_callable=AsyncMock) as mock:
        mock.side_effect = LLMClientError("Test LLM error")
        yield mock


@pytest.fixture
def sample_request_data():
    """
    Sample request data for testing
    """
    return {
        "query": "What's the market outlook for AAPL?",
        "context": {
            "prices": "AAPL: $190.25 (+1.2%), MSFT: $420.10 (-0.5%)",
            "news": "Apple announces new product line. Microsoft releases quarterly earnings.",
            "chunks": "Apple's revenue increased by 10% YoY.",
            "analysis": "PE Ratio: 30.5, EPS: 6.24"
        }
    }
