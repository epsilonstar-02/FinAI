import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import os
import sys
from pathlib import Path
import io

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents.voice_agent.main import app
from agents.voice_agent.stt_client import STTClientError
from agents.voice_agent.tts_client import TTSClientError


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app
    """
    return TestClient(app)


@pytest.fixture
def mock_stt_transcribe():
    """
    Mock the transcribe method to avoid making actual STT API calls
    """
    with patch("agents.voice_agent.main.stt_client.transcribe", new_callable=AsyncMock) as mock:
        mock.return_value = ("This is a mock transcription.", 0.95)
        yield mock


@pytest.fixture
def mock_stt_transcribe_error():
    """
    Mock the transcribe method to simulate an error
    """
    with patch("agents.voice_agent.main.stt_client.transcribe", new_callable=AsyncMock) as mock:
        mock.side_effect = STTClientError("Test STT error")
        yield mock


@pytest.fixture
def mock_tts_synthesize():
    """
    Mock the synthesize method to avoid making actual TTS API calls
    """
    with patch("agents.voice_agent.main.tts_client.synthesize", new_callable=AsyncMock) as mock:
        # Create dummy audio bytes for testing
        mock.return_value = b"mock audio bytes"
        yield mock


@pytest.fixture
def mock_tts_synthesize_error():
    """
    Mock the synthesize method to simulate an error
    """
    with patch("agents.voice_agent.main.tts_client.synthesize", new_callable=AsyncMock) as mock:
        mock.side_effect = TTSClientError("Test TTS error")
        yield mock


@pytest.fixture
def sample_audio_file():
    """
    Create a sample audio file for testing
    """
    return io.BytesIO(b"mock audio content")


@pytest.fixture
def sample_tts_request():
    """
    Sample TTS request data for testing
    """
    return {
        "text": "This is a test text to be converted to speech.",
        "voice": "default",
        "speed": 1.0
    }
