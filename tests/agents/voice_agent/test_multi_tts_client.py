"""
Unit tests for the multi_tts_client module of the Voice Agent.
"""
import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import io
import asyncio
import json

# Add parent directory to path to import the agent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from agents.voice_agent.multi_tts_client import (
    MultiTTSClient,
    get_multi_tts_client,
    synthesize_speech,
    TTSClientError,
    ProviderError,
    AllProvidersFailedError
)
from agents.voice_agent.config import settings, TTSProvider


class TestMultiTTSClient(unittest.TestCase):
    """Test cases for the MultiTTSClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary output directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_output.wav")
        
        # Create patches for various TTS provider modules
        self.patches = []
        
        # Patch pyttsx3
        self.pyttsx3_patch = patch('agents.voice_agent.multi_tts_client.pyttsx3')
        self.mock_pyttsx3 = self.pyttsx3_patch.start()
        self.patches.append(self.pyttsx3_patch)
        
        # Patch gTTS
        self.gtts_patch = patch('agents.voice_agent.multi_tts_client.gTTS')
        self.mock_gtts = self.gtts_patch.start()
        self.patches.append(self.gtts_patch)
        
        # Patch silero
        self.silero_patch = patch('agents.voice_agent.multi_tts_client.torch')
        self.mock_torch = self.silero_patch.start()
        self.patches.append(self.silero_patch)
        
        # Patch is_provider_available to make PYTTSX3, GTTS, and SILERO available
        # (these are our open-source options)
        self.is_provider_available_patch = patch('agents.voice_agent.config.Settings.is_provider_available')
        self.mock_is_provider_available = self.is_provider_available_patch.start()
        self.mock_is_provider_available.side_effect = lambda p: p in [
            TTSProvider.PYTTSX3, 
            TTSProvider.GTTS, 
            TTSProvider.SILERO
        ]
        self.patches.append(self.is_provider_available_patch)
        
        # Patch os.path.exists to make output file seem to exist
        self.exists_patch = patch('os.path.exists', return_value=True)
        self.mock_exists = self.exists_patch.start()
        self.patches.append(self.exists_patch)
        
        # Initialize the client
        self.client = MultiTTSClient()
        
        # Setup pyttsx3 mock
        mock_engine = MagicMock()
        self.mock_pyttsx3.init.return_value = mock_engine
        
        # Setup Silero mock
        mock_model = MagicMock()
        mock_model.return_value = b"silero audio data"
        self.mock_torch.hub.load.return_value = (mock_model, MagicMock(), MagicMock(), MagicMock())
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
        
        # Stop all patches
        for patch in self.patches:
            patch.stop()
    
    def test_initialization(self):
        """Test that the client initializes correctly."""
        # Check that the available providers are set correctly
        self.assertEqual(set(self.client.available_providers), {
            TTSProvider.PYTTSX3, 
            TTSProvider.GTTS, 
            TTSProvider.SILERO
        })
        
        # Check that provider errors dict is empty initially
        self.assertEqual(self.client.provider_errors, {})
        
        # Check that the cache directory is set
        self.assertIsNotNone(self.client.cache_dir)
    
    async def test_synthesize_pyttsx3(self):
        """Test synthesizing with pyttsx3 (free and offline)."""
        # Call synthesize with pyttsx3 provider
        with patch('builtins.open', mock_open()) as mock_file:
            result = await self.client.synthesize("Test text", self.output_path, provider=TTSProvider.PYTTSX3)
        
        # Verify that the result is the output path
        self.assertEqual(result, self.output_path)
        
        # Verify that pyttsx3 was used
        self.mock_pyttsx3.init.assert_called_once()
        self.mock_pyttsx3.init.return_value.say.assert_called_with("Test text")
        self.mock_pyttsx3.init.return_value.save_to_file.assert_called_with("Test text", self.output_path)
        self.mock_pyttsx3.init.return_value.runAndWait.assert_called()
    
    async def test_synthesize_gtts(self):
        """Test synthesizing with gTTS (free)."""
        # Call synthesize with gTTS provider
        result = await self.client.synthesize("Test text", self.output_path, provider=TTSProvider.GTTS)
        
        # Verify that the result is the output path
        self.assertEqual(result, self.output_path)
        
        # Verify that gTTS was used
        self.mock_gtts.assert_called_once_with("Test text", lang="en")
        self.mock_gtts.return_value.save.assert_called_once_with(self.output_path)
    
    async def test_synthesize_silero(self):
        """Test synthesizing with Silero (free and open-source)."""
        # Call synthesize with Silero provider
        with patch('builtins.open', mock_open()) as mock_file:
            result = await self.client.synthesize("Test text", self.output_path, provider=TTSProvider.SILERO)
        
        # Verify that the result is the output path
        self.assertEqual(result, self.output_path)
        
        # Verify that Silero was used
        self.mock_torch.hub.load.assert_called_once()
        
        # Verify that the output file was written
        mock_file.assert_called_with(self.output_path, 'wb')
        mock_file.return_value.write.assert_called_with(b"silero audio data")
    
    async def test_provider_fallback(self):
        """Test fallback between providers."""
        # Make pyttsx3 fail
        self.mock_pyttsx3.init.side_effect = Exception("pyttsx3 failed")
        
        # Call synthesize with default providers (should fallback to gTTS)
        result = await self.client.synthesize("Test text", self.output_path)
        
        # Verify that the result is the output path
        self.assertEqual(result, self.output_path)
        
        # Verify that both providers were attempted
        self.mock_pyttsx3.init.assert_called_once()
        self.mock_gtts.assert_called_once()
        
        # Verify that the error was recorded
        self.assertIn(TTSProvider.PYTTSX3.value, self.client.provider_errors)
    
    async def test_all_providers_fail(self):
        """Test behavior when all providers fail."""
        # Make all providers fail
        self.mock_pyttsx3.init.side_effect = Exception("pyttsx3 failed")
        self.mock_gtts.side_effect = Exception("gTTS failed")
        self.mock_torch.hub.load.side_effect = Exception("Silero failed")
        
        # Call synthesize and expect an AllProvidersFailedError
        with self.assertRaises(AllProvidersFailedError):
            await self.client.synthesize("Test text", self.output_path)
        
        # Verify that all providers were attempted
        self.mock_pyttsx3.init.assert_called_once()
        self.mock_gtts.assert_called_once()
        self.mock_torch.hub.load.assert_called_once()
        
        # Verify that errors were recorded for all providers
        self.assertIn(TTSProvider.PYTTSX3.value, self.client.provider_errors)
        self.assertIn(TTSProvider.GTTS.value, self.client.provider_errors)
        self.assertIn(TTSProvider.SILERO.value, self.client.provider_errors)
    
    async def test_caching(self):
        """Test that synthesized speech is cached."""
        # Enable caching
        self.client.use_cache = True
        
        # First call should synthesize and cache
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('agents.voice_agent.multi_tts_client.hashlib.md5') as mock_md5:
                # Create a mock MD5 hash
                mock_hash = MagicMock()
                mock_hash.hexdigest.return_value = "test_hash"
                mock_md5.return_value = mock_hash
                
                # First call should use provider
                result1 = await self.client.synthesize("Test text", self.output_path, provider=TTSProvider.PYTTSX3)
                
                # Verify that the engine was initialized
                self.mock_pyttsx3.init.assert_called_once()
                
                # Reset mocks for second call
                self.mock_pyttsx3.reset_mock()
                
                # Second call with same text should use cache
                with patch('shutil.copyfile') as mock_copyfile:
                    # Make exists return True for cache file
                    with patch('os.path.exists', side_effect=lambda p: True):
                        result2 = await self.client.synthesize("Test text", self.output_path, provider=TTSProvider.PYTTSX3)
                
                # Verify that the engine was not initialized again
                self.mock_pyttsx3.init.assert_not_called()
                
                # Verify that copyfile was called
                mock_copyfile.assert_called_once()
    
    def test_singleton_pattern(self):
        """Test that get_multi_tts_client returns a singleton instance."""
        with patch('agents.voice_agent.multi_tts_client.MultiTTSClient') as mock_client:
            # Reset the singleton first
            import agents.voice_agent.multi_tts_client
            agents.voice_agent.multi_tts_client._multi_tts_client = None
            
            # First call should create a new instance
            first = get_multi_tts_client()
            mock_client.assert_called_once()
            
            # Reset the mock to check if it's called again
            mock_client.reset_mock()
            
            # Second call should return the existing instance
            second = get_multi_tts_client()
            mock_client.assert_not_called()
            
            # Both calls should return the same instance
            self.assertEqual(first, second)
    
    async def test_module_level_synthesize_speech(self):
        """Test the module-level synthesize_speech function."""
        with patch('agents.voice_agent.multi_tts_client.get_multi_tts_client') as mock_get_client:
            # Create mock client
            mock_client = MagicMock()
            mock_client.synthesize = AsyncMock(return_value="test_output.wav")
            mock_get_client.return_value = mock_client
            
            # Call the module-level synthesize_speech function
            result = await synthesize_speech("Test text", self.output_path, provider=TTSProvider.PYTTSX3)
            
            # Verify that the result is correct
            self.assertEqual(result, "test_output.wav")
            
            # Verify that get_multi_tts_client was called
            mock_get_client.assert_called_once()
            
            # Verify that synthesize was called on the client
            mock_client.synthesize.assert_called_once_with("Test text", self.output_path, TTSProvider.PYTTSX3)


# Helper function to run async tests
def run_async_test(coroutine):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Modify the test class to run async tests
for name, method in list(TestMultiTTSClient.__dict__.items()):
    if name.startswith('test_') and asyncio.iscoroutinefunction(method):
        setattr(TestMultiTTSClient, name, lambda self, method=method: run_async_test(method(self)))


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
