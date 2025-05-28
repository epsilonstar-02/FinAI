"""
Unit tests for the FastAPI endpoints of the Voice Agent.
"""
import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import json
from fastapi.testclient import TestClient
from fastapi import UploadFile

# Add parent directory to path to import the agent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from agents.voice_agent.main import app
from agents.voice_agent.models import (
    STTResponse,
    TTSRequest,
    TTSResponse,
    HealthResponse
)
from agents.voice_agent.config import STTProvider, TTSProvider


class TestVoiceAgentEndpoints(unittest.TestCase):
    """Test cases for the Voice Agent API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test client
        self.client = TestClient(app)
        
        # Create temporary files
        self.temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_audio.write(b"dummy audio data")
        self.temp_audio.close()
        
        self.temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_output.close()
        
        # Patch the transcribe_audio function
        self.transcribe_audio_patch = patch('agents.voice_agent.main.transcribe_audio')
        self.mock_transcribe_audio = self.transcribe_audio_patch.start()
        self.mock_transcribe_audio.return_value = "Transcribed speech"
        
        # Patch the synthesize_speech function
        self.synthesize_speech_patch = patch('agents.voice_agent.main.synthesize_speech')
        self.mock_synthesize_speech = self.synthesize_speech_patch.start()
        self.mock_synthesize_speech.return_value = self.temp_output.name
        
        # Patch the get_available_stt_providers function
        self.get_available_stt_providers_patch = patch('agents.voice_agent.main.get_available_stt_providers')
        self.mock_get_available_stt_providers = self.get_available_stt_providers_patch.start()
        self.mock_get_available_stt_providers.return_value = {
            "whisper_local": "Whisper Local",
            "speechrecognition": "SpeechRecognition",
            "vosk": "Vosk"
        }
        
        # Patch the get_available_tts_providers function
        self.get_available_tts_providers_patch = patch('agents.voice_agent.main.get_available_tts_providers')
        self.mock_get_available_tts_providers = self.get_available_tts_providers_patch.start()
        self.mock_get_available_tts_providers.return_value = {
            "pyttsx3": "pyttsx3",
            "gtts": "Google Text-to-Speech",
            "silero": "Silero"
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.transcribe_audio_patch.stop()
        self.synthesize_speech_patch.stop()
        self.get_available_stt_providers_patch.stop()
        self.get_available_tts_providers_patch.stop()
        
        # Remove temporary files
        os.unlink(self.temp_audio.name)
        os.unlink(self.temp_output.name)
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("stt_providers", data)
        self.assertIn("tts_providers", data)
        self.assertIn("timestamp", data)
    
    def test_get_stt_providers(self):
        """Test the STT providers endpoint."""
        response = self.client.get("/stt/providers")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("providers", data)
        self.assertEqual(len(data["providers"]), 3)
        self.assertIn("whisper_local", data["providers"])
        self.assertIn("speechrecognition", data["providers"])
        self.assertIn("vosk", data["providers"])
    
    def test_get_tts_providers(self):
        """Test the TTS providers endpoint."""
        response = self.client.get("/tts/providers")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("providers", data)
        self.assertEqual(len(data["providers"]), 3)
        self.assertIn("pyttsx3", data["providers"])
        self.assertIn("gtts", data["providers"])
        self.assertIn("silero", data["providers"])
    
    def test_speech_to_text(self):
        """Test the speech-to-text endpoint."""
        # Create file for upload
        with open(self.temp_audio.name, 'rb') as f:
            response = self.client.post(
                "/stt",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "Transcribed speech")
        self.assertEqual(data["provider"], "whisper_local")  # Default provider
        
        # Verify transcribe_audio was called
        self.mock_transcribe_audio.assert_called_once()
    
    def test_speech_to_text_with_provider(self):
        """Test the speech-to-text endpoint with a specific provider."""
        # Create file for upload
        with open(self.temp_audio.name, 'rb') as f:
            response = self.client.post(
                "/stt?provider=vosk",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "Transcribed speech")
        self.assertEqual(data["provider"], "vosk")
        
        # Verify transcribe_audio was called with the correct provider
        self.mock_transcribe_audio.assert_called_with(
            unittest.mock.ANY,  # We don't care about the exact temp file path
            STTProvider.VOSK
        )
    
    def test_text_to_speech(self):
        """Test the text-to-speech endpoint."""
        # Create request payload
        payload = {
            "text": "Text to synthesize",
            "provider": "pyttsx3"
        }
        
        # Make request to TTS endpoint
        response = self.client.post("/tts", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Verify that the response contains audio data
        self.assertEqual(response.headers["content-type"], "audio/wav")
        
        # Verify synthesize_speech was called with the correct arguments
        self.mock_synthesize_speech.assert_called_with(
            "Text to synthesize",
            unittest.mock.ANY,  # We don't care about the exact temp file path
            TTSProvider.PYTTSX3
        )
    
    def test_text_to_speech_with_default_provider(self):
        """Test the text-to-speech endpoint with the default provider."""
        # Create request payload
        payload = {
            "text": "Text to synthesize"
        }
        
        # Make request to TTS endpoint
        response = self.client.post("/tts", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Verify that the response contains audio data
        self.assertEqual(response.headers["content-type"], "audio/wav")
        
        # Verify synthesize_speech was called with the default provider
        self.mock_synthesize_speech.assert_called_with(
            "Text to synthesize",
            unittest.mock.ANY,  # We don't care about the exact temp file path
            TTSProvider.PYTTSX3  # Default provider set in our configuration
        )
    
    def test_error_handling_stt(self):
        """Test error handling in STT endpoint."""
        # Make transcribe_audio raise an exception
        self.mock_transcribe_audio.side_effect = ValueError("Test error")
        
        # Create file for upload
        with open(self.temp_audio.name, 'rb') as f:
            response = self.client.post(
                "/stt",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        
        # Check response
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Test error", data["detail"])
    
    def test_error_handling_tts(self):
        """Test error handling in TTS endpoint."""
        # Make synthesize_speech raise an exception
        self.mock_synthesize_speech.side_effect = ValueError("Test error")
        
        # Create request payload
        payload = {
            "text": "Text to synthesize"
        }
        
        # Make request to TTS endpoint
        response = self.client.post("/tts", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Test error", data["detail"])


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
