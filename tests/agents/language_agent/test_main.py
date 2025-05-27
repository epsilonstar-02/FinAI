import pytest
from fastapi import HTTPException
import json
from unittest.mock import patch, AsyncMock

from agents.language_agent.models import GenerateRequest, GenerateResponse


class TestLanguageAgentAPI:
    """Tests for the Language Agent API endpoints"""

    def test_health_endpoint(self, client):
        """Test the health endpoint returns correct status"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent": "Language Agent"}

    def test_generate_endpoint_success(self, client, mock_generate_text, sample_request_data):
        """Test successful text generation"""
        response = client.post("/generate", json=sample_request_data)
        
        # Check response
        assert response.status_code == 200
        assert "text" in response.json()
        assert response.json()["text"] == "This is a mock response from the language model."
        
        # Verify mock was called with expected arguments
        mock_generate_text.assert_called_once()
        # Extract the prompt string passed to generate_text
        prompt_arg = mock_generate_text.call_args[0][0]
        
        # Verify the prompt contains key elements from the request
        assert sample_request_data["query"] in prompt_arg
        assert sample_request_data["context"]["prices"] in prompt_arg
        assert sample_request_data["context"]["news"] in prompt_arg

    def test_generate_endpoint_llm_error(self, client, mock_generate_text_error, sample_request_data):
        """Test LLM error handling"""
        response = client.post("/generate", json=sample_request_data)
        
        # Check response
        assert response.status_code == 502
        assert "detail" in response.json()
        assert "Test LLM error" in response.json()["detail"]

    def test_generate_endpoint_validation_error(self, client):
        """Test request validation error handling"""
        # Missing required fields
        invalid_data = {"query": "Test query"}  # Missing context
        
        response = client.post("/generate", json=invalid_data)
        
        # Check response
        assert response.status_code == 422  # Unprocessable Entity
        assert "detail" in response.json()

    def test_generate_endpoint_empty_context(self, client, mock_generate_text):
        """Test with empty context dictionary"""
        request_data = {
            "query": "What's the market outlook?",
            "context": {}  # Empty but valid context
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should still work
        assert response.status_code == 200
        assert "text" in response.json()
