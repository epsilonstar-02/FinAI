"""
Unit tests for the multi_stt_client module of the Voice Agent.
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

from agents.voice_agent.multi_stt_client import (
    MultiSTTClient,
    get_multi_stt_client,
    transcribe_audio,
    STTClientError,
    ProviderError,
    AllProvidersFailedError
)
from agents.voice_agent.config import settings, STTProvider


class TestMultiSTTClient(unittest.TestCase):
    """Test cases for the MultiSTTClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary audio file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.write(b"dummy audio data")
        self.temp_file.close()
        
        # Create patches for various STT provider modules
        self.patches = []
        
        # Patch whisper
        self.whisper_patch = patch('agents.voice_agent.multi_stt_client.whisper')
        self.mock_whisper = self.whisper_patch.start()
        self.patches.append(self.whisper_patch)
        
        # Patch speech_recognition
        self.sr_patch = patch('agents.voice_agent.multi_stt_client.sr')
        self.mock_sr = self.sr_patch.start()
        self.patches.append(self.sr_patch)
        
        # Patch vosk
        self.vosk_patch = patch('agents.voice_agent.multi_stt_client.Model')
        self.mock_vosk_model = self.vosk_patch.start()
        self.patches.append(self.vosk_patch)
        
        self.vosk_recognizer_patch = patch('agents.voice_agent.multi_stt_client.KaldiRecognizer')
        self.mock_vosk_recognizer = self.vosk_recognizer_patch.start()
        self.patches.append(self.vosk_recognizer_patch)
        
        # Patch is_provider_available to make WHISPER_LOCAL, SPEECHRECOGNITION, and VOSK available
        # (these are our open-source options)
        self.is_provider_available_patch = patch('agents.voice_agent.config.Settings.is_provider_available')
        self.mock_is_provider_available = self.is_provider_available_patch.start()
        self.mock_is_provider_available.side_effect = lambda p: p in [
            STTProvider.WHISPER_LOCAL, 
            STTProvider.SPEECHRECOGNITION, 
            STTProvider.VOSK
        ]
        self.patches.append(self.is_provider_available_patch)
        
        # Initialize the client
        self.client = MultiSTTClient()
        
        # Mock STT responses
        self.mock_whisper_response = "This is a whisper transcription."
        self.mock_sr_response = "This is a speech recognition transcription."
        self.mock_vosk_response = "This is a vosk transcription."
        
        # Set up whisper mock response
        mock_whisper_result = MagicMock()
        mock_whisper_result.text = self.mock_whisper_response
        self.mock_whisper.load_model.return_value = MagicMock()
        self.mock_whisper.load_model.return_value.transcribe.return_value = mock_whisper_result
        
        # Set up speech_recognition mock response
        mock_recognizer = MagicMock()
        mock_recognizer.recognize_sphinx.return_value = self.mock_sr_response
        self.mock_sr.Recognizer.return_value = mock_recognizer
        
        # Set up vosk mock response
        mock_vosk_instance = MagicMock()
        mock_vosk_instance.AcceptWaveform.return_value = True
        mock_vosk_instance.Result.return_value = json.dumps({"text": self.mock_vosk_response})
        self.mock_vosk_recognizer.return_value = mock_vosk_instance
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary file
        os.unlink(self.temp_file.name)
        
        # Stop all patches
        for patch in self.patches:
            patch.stop()
    
    def test_initialization(self):
        """Test that the client initializes correctly."""
        # Check that the available providers are set correctly
        self.assertEqual(set(self.client.available_providers), {
            STTProvider.WHISPER_LOCAL, 
            STTProvider.SPEECHRECOGNITION, 
            STTProvider.VOSK
        })
        
        # Check that provider errors dict is empty initially
        self.assertEqual(self.client.provider_errors, {})
    
    async def test_transcribe_whisper_local(self):
        """Test transcribing with local Whisper (free and open-source)."""
        # Call transcribe with Whisper provider
        result = await self.client.transcribe(self.temp_file.name, provider=STTProvider.WHISPER_LOCAL)
        
        # Verify that the result is correct
        self.assertEqual(result, self.mock_whisper_response)
        
        # Verify that Whisper was used
        self.mock_whisper.load_model.assert_called_once()
    
    async def test_transcribe_speechrecognition(self):
        """Test transcribing with SpeechRecognition (free)."""
        # Call transcribe with SpeechRecognition provider
        with patch('builtins.open', mock_open(read_data=b'dummy audio data')):
            result = await self.client.transcribe(self.temp_file.name, provider=STTProvider.SPEECHRECOGNITION)
        
        # Verify that the result is correct
        self.assertEqual(result, self.mock_sr_response)
        
        # Verify that SpeechRecognition was used
        self.mock_sr.Recognizer.assert_called_once()
        self.mock_sr.Recognizer.return_value.recognize_sphinx.assert_called_once()
    
    async def test_transcribe_vosk(self):
        """Test transcribing with Vosk (free and open-source)."""
        # Call transcribe with Vosk provider
        with patch('builtins.open', mock_open(read_data=b'dummy audio data')):
            with patch('wave.open', MagicMock()):
                result = await self.client.transcribe(self.temp_file.name, provider=STTProvider.VOSK)
        
        # Verify that the result is correct
        self.assertEqual(result, self.mock_vosk_response)
        
        # Verify that Vosk was used
        self.mock_vosk_model.assert_called_once()
        self.mock_vosk_recognizer.assert_called_once()
    
    async def test_provider_fallback(self):
        """Test fallback between providers."""
        # Make Whisper fail
        self.mock_whisper.load_model.side_effect = Exception("Whisper failed")
        
        # Call transcribe with default providers (should fallback to SpeechRecognition)
        with patch('builtins.open', mock_open(read_data=b'dummy audio data')):
            result = await self.client.transcribe(self.temp_file.name)
        
        # Verify that the result is correct (from SpeechRecognition)
        self.assertEqual(result, self.mock_sr_response)
        
        # Verify that both providers were attempted
        self.mock_whisper.load_model.assert_called_once()
        self.mock_sr.Recognizer.assert_called_once()
        
        # Verify that the error was recorded
        self.assertIn(STTProvider.WHISPER_LOCAL.value, self.client.provider_errors)
    
    async def test_all_providers_fail(self):
        """Test behavior when all providers fail."""
        # Make all providers fail
        self.mock_whisper.load_model.side_effect = Exception("Whisper failed")
        self.mock_sr.Recognizer.side_effect = Exception("SpeechRecognition failed")
        self.mock_vosk_model.side_effect = Exception("Vosk failed")
        
        # Call transcribe and expect an AllProvidersFailedError
        with self.assertRaises(AllProvidersFailedError):
            await self.client.transcribe(self.temp_file.name)
        
        # Verify that all providers were attempted
        self.mock_whisper.load_model.assert_called_once()
        self.mock_sr.Recognizer.assert_called_once()
        self.mock_vosk_model.assert_called_once()
        
        # Verify that errors were recorded for all providers
        self.assertIn(STTProvider.WHISPER_LOCAL.value, self.client.provider_errors)
        self.assertIn(STTProvider.SPEECHRECOGNITION.value, self.client.provider_errors)
        self.assertIn(STTProvider.VOSK.value, self.client.provider_errors)
    
    def test_singleton_pattern(self):
        """Test that get_multi_stt_client returns a singleton instance."""
        with patch('agents.voice_agent.multi_stt_client.MultiSTTClient') as mock_client:
            # Reset the singleton first
            import agents.voice_agent.multi_stt_client
            agents.voice_agent.multi_stt_client._multi_stt_client = None
            
            # First call should create a new instance
            first = get_multi_stt_client()
            mock_client.assert_called_once()
            
            # Reset the mock to check if it's called again
            mock_client.reset_mock()
            
            # Second call should return the existing instance
            second = get_multi_stt_client()
            mock_client.assert_not_called()
            
            # Both calls should return the same instance
            self.assertEqual(first, second)
    
    async def test_module_level_transcribe_audio(self):
        """Test the module-level transcribe_audio function."""
        with patch('agents.voice_agent.multi_stt_client.get_multi_stt_client') as mock_get_client:
            # Create mock client
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(return_value="Test transcription")
            mock_get_client.return_value = mock_client
            
            # Call the module-level transcribe_audio function
            result = await transcribe_audio(self.temp_file.name, provider=STTProvider.WHISPER_LOCAL)
            
            # Verify that the result is correct
            self.assertEqual(result, "Test transcription")
            
            # Verify that get_multi_stt_client was called
            mock_get_client.assert_called_once()
            
            # Verify that transcribe was called on the client
            mock_client.transcribe.assert_called_once_with(self.temp_file.name, STTProvider.WHISPER_LOCAL)


# Helper function to run async tests
def run_async_test(coroutine):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Modify the test class to run async tests
for name, method in list(TestMultiSTTClient.__dict__.items()):
    if name.startswith('test_') and asyncio.iscoroutinefunction(method):
        setattr(TestMultiSTTClient, name, lambda self, method=method: run_async_test(method(self)))


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
