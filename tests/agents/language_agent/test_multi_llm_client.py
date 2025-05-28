"""
Unit tests for the multi_llm_client module of the Language Agent.
"""
import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import json
import asyncio

# Add parent directory to path to import the agent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from agents.language_agent.multi_llm_client import (
    MultiLLMClient,
    get_multi_llm_client,
    generate_text,
    LLMClientError,
    ProviderError,
    AllProvidersFailedError
)
from agents.language_agent.config import settings, LLMProvider


class TestMultiLLMClient(unittest.TestCase):
    """Test cases for the MultiLLMClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patches for various LLM provider modules
        self.patches = []
        
        # Patch Google Generative AI
        self.genai_patch = patch('agents.language_agent.multi_llm_client.genai')
        self.mock_genai = self.genai_patch.start()
        self.patches.append(self.genai_patch)
        
        # Patch OpenAI
        self.openai_patch = patch('agents.language_agent.multi_llm_client.openai')
        self.mock_openai = self.openai_patch.start()
        self.patches.append(self.openai_patch)
        
        # Patch LangChain LlamaCpp
        self.llama_patch = patch('agents.language_agent.multi_llm_client.LlamaCpp')
        self.mock_llama = self.llama_patch.start()
        self.patches.append(self.llama_patch)
        
        # Patch HuggingFace
        self.hf_patch = patch('agents.language_agent.multi_llm_client.InferenceClient')
        self.mock_hf = self.hf_patch.start()
        self.patches.append(self.hf_patch)
        
        # Patch is_provider_available to make LLAMA and HUGGINGFACE available
        # (these are our open-source options)
        self.is_provider_available_patch = patch('agents.language_agent.config.Settings.is_provider_available')
        self.mock_is_provider_available = self.is_provider_available_patch.start()
        self.mock_is_provider_available.side_effect = lambda p: p in [LLMProvider.LLAMA, LLMProvider.HUGGINGFACE]
        self.patches.append(self.is_provider_available_patch)
        
        # Initialize the client
        self.client = MultiLLMClient()
        
        # Mock LLM responses
        self.mock_llama_response = "This is a response from Llama."
        self.mock_hf_response = "This is a response from HuggingFace."
        
        # Set up Llama mock response
        self.mock_llama_instance = MagicMock()
        self.mock_llama_instance.invoke.return_value = self.mock_llama_response
        self.mock_llama.return_value = self.mock_llama_instance
        
        # Set up HuggingFace mock response
        self.mock_hf_instance = MagicMock()
        self.mock_hf_instance.text_generation.return_value = self.mock_hf_response
        self.mock_hf.return_value = self.mock_hf_instance
    
    def tearDown(self):
        """Clean up after tests."""
        for patch in self.patches:
            patch.stop()
    
    def test_initialization(self):
        """Test that the client initializes correctly."""
        # Check that the available providers are set correctly
        self.assertEqual(set(self.client.available_providers), {LLMProvider.LLAMA, LLMProvider.HUGGINGFACE})
        
        # Check that provider errors dict is empty initially
        self.assertEqual(self.client.provider_errors, {})
    
    async def test_generate_text_llama(self):
        """Test generating text with Llama (free and open-source)."""
        # Call generate_text with Llama provider
        result = await self.client.generate_text("Test prompt", provider=LLMProvider.LLAMA)
        
        # Verify that the result is correct
        self.assertEqual(result, self.mock_llama_response)
        
        # Verify that Llama was used
        self.mock_llama.assert_called_once()
        self.mock_llama_instance.invoke.assert_called_once_with("Test prompt")
    
    async def test_generate_text_huggingface(self):
        """Test generating text with HuggingFace (free and open-source)."""
        # Call generate_text with HuggingFace provider
        result = await self.client.generate_text("Test prompt", provider=LLMProvider.HUGGINGFACE)
        
        # Verify that the result is correct
        self.assertEqual(result, self.mock_hf_response)
        
        # Verify that HuggingFace was used
        self.mock_hf.assert_called_once()
        self.mock_hf_instance.text_generation.assert_called_once()
    
    async def test_provider_fallback(self):
        """Test fallback between providers."""
        # Make Llama fail
        self.mock_llama_instance.invoke.side_effect = Exception("Llama failed")
        
        # Call generate_text with default providers (should fallback to HuggingFace)
        result = await self.client.generate_text("Test prompt")
        
        # Verify that the result is correct (from HuggingFace)
        self.assertEqual(result, self.mock_hf_response)
        
        # Verify that both providers were attempted
        self.mock_llama.assert_called_once()
        self.mock_llama_instance.invoke.assert_called_once()
        self.mock_hf.assert_called_once()
        self.mock_hf_instance.text_generation.assert_called_once()
        
        # Verify that the error was recorded
        self.assertIn(LLMProvider.LLAMA.value, self.client.provider_errors)
    
    async def test_all_providers_fail(self):
        """Test behavior when all providers fail."""
        # Make all providers fail
        self.mock_llama_instance.invoke.side_effect = Exception("Llama failed")
        self.mock_hf_instance.text_generation.side_effect = Exception("HuggingFace failed")
        
        # Call generate_text and expect an AllProvidersFailedError
        with self.assertRaises(AllProvidersFailedError):
            await self.client.generate_text("Test prompt")
        
        # Verify that both providers were attempted
        self.mock_llama.assert_called_once()
        self.mock_llama_instance.invoke.assert_called_once()
        self.mock_hf.assert_called_once()
        self.mock_hf_instance.text_generation.assert_called_once()
        
        # Verify that errors were recorded for both providers
        self.assertIn(LLMProvider.LLAMA.value, self.client.provider_errors)
        self.assertIn(LLMProvider.HUGGINGFACE.value, self.client.provider_errors)
    
    def test_singleton_pattern(self):
        """Test that get_multi_llm_client returns a singleton instance."""
        with patch('agents.language_agent.multi_llm_client.MultiLLMClient') as mock_client:
            # Reset the singleton first
            import agents.language_agent.multi_llm_client
            agents.language_agent.multi_llm_client._multi_llm_client = None
            
            # First call should create a new instance
            first = get_multi_llm_client()
            mock_client.assert_called_once()
            
            # Reset the mock to check if it's called again
            mock_client.reset_mock()
            
            # Second call should return the existing instance
            second = get_multi_llm_client()
            mock_client.assert_not_called()
            
            # Both calls should return the same instance
            self.assertEqual(first, second)
    
    async def test_module_level_generate_text(self):
        """Test the module-level generate_text function."""
        with patch('agents.language_agent.multi_llm_client.get_multi_llm_client') as mock_get_client:
            # Create mock client
            mock_client = MagicMock()
            mock_client.generate_text = AsyncMock(return_value="Test response")
            mock_get_client.return_value = mock_client
            
            # Call the module-level generate_text function
            result = await generate_text("Test prompt", provider=LLMProvider.LLAMA)
            
            # Verify that the result is correct
            self.assertEqual(result, "Test response")
            
            # Verify that get_multi_llm_client was called
            mock_get_client.assert_called_once()
            
            # Verify that generate_text was called on the client
            mock_client.generate_text.assert_called_once_with("Test prompt", LLMProvider.LLAMA)


# Helper function to run async tests
def run_async_test(coroutine):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Modify the test class to run async tests
for name, method in list(TestMultiLLMClient.__dict__.items()):
    if name.startswith('test_') and asyncio.iscoroutinefunction(method):
        setattr(TestMultiLLMClient, name, lambda self, method=method: run_async_test(method(self)))


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
