import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio

from agents.language_agent.llm_client import generate_text, LLMClientError
from agents.language_agent.config import settings


class TestLLMClient:
    """Tests for the Language Agent LLM Client"""

    @pytest.mark.asyncio
    async def test_generate_text_success(self):
        """Test successful text generation"""
        # Mock response from Google Generative AI
        mock_response = MagicMock()
        mock_response.text = "This is a successful test response."
        
        # Mock the chat session
        mock_chat = AsyncMock()
        mock_chat.send_message_async.return_value = mock_response
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        
        # Patch the GenerativeModel
        with patch("agents.language_agent.llm_client.genai.GenerativeModel", return_value=mock_model):
            result = await generate_text("Test prompt")
            
            # Assert correct response
            assert result == "This is a successful test response."
            
            # Verify mocks were called with expected args
            mock_model.start_chat.assert_called_once()
            mock_chat.send_message_async.assert_called_once()
            
            # Verify generation config was passed
            args, kwargs = mock_chat.send_message_async.call_args
            assert "Test prompt" == args[0]
            assert "generation_config" in kwargs
            assert kwargs["generation_config"]["temperature"] == 0.2
            assert kwargs["generation_config"]["top_k"] == 40

    @pytest.mark.asyncio
    async def test_generate_text_api_error(self):
        """Test error handling when the API call fails"""
        # Mock chat session that raises an exception
        mock_chat = AsyncMock()
        mock_chat.send_message_async.side_effect = Exception("API error")
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        
        # Patch the GenerativeModel
        with patch("agents.language_agent.llm_client.genai.GenerativeModel", return_value=mock_model):
            with pytest.raises(LLMClientError) as exc_info:
                await generate_text("Test prompt")
            
            # Check the error message
            assert "Error generating text: API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_text_timeout(self):
        """Test timeout handling"""
        # Mock chat session that times out
        mock_chat = AsyncMock()
        mock_chat.send_message_async.side_effect = asyncio.TimeoutError()
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        
        # Patch the GenerativeModel
        with patch("agents.language_agent.llm_client.genai.GenerativeModel", return_value=mock_model):
            with pytest.raises(LLMClientError) as exc_info:
                await generate_text("Test prompt")
            
            # Check the error message
            assert f"Request timed out after {settings.TIMEOUT} seconds" in str(exc_info.value)
