import pytest
from fastapi import HTTPException
import json
from unittest.mock import patch, AsyncMock
import io

from agents.voice_agent.models import STTResponse, TTSRequest


class TestVoiceAgentAPI:
    """Tests for the Voice Agent API endpoints"""

    def test_health_endpoint(self, client):
        """Test the health endpoint returns correct status"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent": "Voice Agent"}

    def test_stt_endpoint_success(self, client, mock_stt_transcribe, sample_audio_file):
        """Test successful speech-to-text conversion"""
        # Create multipart form with audio file
        files = {"file": ("test_audio.wav", sample_audio_file, "audio/wav")}
        
        response = client.post("/stt", files=files)
        
        # Check response
        assert response.status_code == 200
        assert "text" in response.json()
        assert "confidence" in response.json()
        assert response.json()["text"] == "This is a mock transcription."
        assert response.json()["confidence"] == 0.95
        
        # Verify mock was called
        mock_stt_transcribe.assert_called_once()

    def test_stt_endpoint_error(self, client, mock_stt_transcribe_error, sample_audio_file):
        """Test STT error handling"""
        # Create multipart form with audio file
        files = {"file": ("test_audio.wav", sample_audio_file, "audio/wav")}
        
        response = client.post("/stt", files=files)
        
        # Check response
        assert response.status_code == 502
        assert "detail" in response.json()
        assert "Test STT error" in response.json()["detail"]

    def test_tts_endpoint_success(self, client, mock_tts_synthesize, sample_tts_request):
        """Test successful text-to-speech conversion"""
        response = client.post("/tts", json=sample_tts_request)
        
        # Check response
        assert response.status_code == 200
        assert response.content == b"mock audio bytes"
        assert response.headers["Content-Type"] == "audio/mpeg"
        assert "attachment; filename=speech.mp3" in response.headers["Content-Disposition"]
        
        # Verify mock was called with expected arguments
        mock_tts_synthesize.assert_called_once_with(
            text=sample_tts_request["text"],
            voice=sample_tts_request["voice"],
            speed=sample_tts_request["speed"]
        )

    def test_tts_endpoint_error(self, client, mock_tts_synthesize_error, sample_tts_request):
        """Test TTS error handling"""
        response = client.post("/tts", json=sample_tts_request)
        
        # Check response
        assert response.status_code == 502
        assert "detail" in response.json()
        assert "Test TTS error" in response.json()["detail"]

    def test_tts_endpoint_validation_error(self, client):
        """Test request validation error handling"""
        # Missing required fields
        invalid_data = {}  # Missing text field
        
        response = client.post("/tts", json=invalid_data)
        
        # Check response
        assert response.status_code == 422  # Unprocessable Entity
        assert "detail" in response.json()

    def test_tts_endpoint_custom_params(self, client, mock_tts_synthesize):
        """Test TTS with custom voice and speed parameters"""
        custom_request = {
            "text": "This is a custom voice test.",
            "voice": "female",
            "speed": 1.5
        }
        
        response = client.post("/tts", json=custom_request)
        
        # Check response
        assert response.status_code == 200
        
        # Verify mock was called with custom parameters
        mock_tts_synthesize.assert_called_once_with(
            text=custom_request["text"],
            voice=custom_request["voice"],
            speed=custom_request["speed"]
        )
