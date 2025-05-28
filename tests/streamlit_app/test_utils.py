import io
import os
import pytest
import requests
import requests_mock
from fpdf import FPDF

# Update path to import utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from streamlit_app.utils import call_orchestrator, call_stt, call_tts, generate_pdf


@pytest.fixture
def mock_env(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("ORCH_URL", "http://test-orchestrator:8001")
    monkeypatch.setenv("VOICE_URL", "http://test-voice:8006")


@pytest.fixture
def sample_audio_bytes():
    """Create a small sample of audio bytes for testing."""
    return b"dummy audio bytes for testing"


def test_call_orchestrator_success(requests_mock):
    """Test call_orchestrator returns parsed JSON when request is successful."""
    mock_response = {"output": "Financial analysis completed successfully"}
    requests_mock.post("http://test-orchestrator:8001/run", json=mock_response)
    
    params = {
        "ORCH_URL": "http://test-orchestrator:8001",
        "mode": "text",
        "news_limit": 3,
        "retrieve_k": 5,
        "include_analysis": True
    }
    
    result = call_orchestrator("What's the latest on AAPL stock?", params)
    
    assert result == mock_response
    assert "output" in result
    assert result["output"] == "Financial analysis completed successfully"


def test_call_orchestrator_error(requests_mock):
    """Test call_orchestrator returns None when request fails."""
    requests_mock.post(
        "http://test-orchestrator:8001/run",
        status_code=500,
        json={"error": "Internal server error"}
    )
    
    params = {
        "ORCH_URL": "http://test-orchestrator:8001",
        "mode": "text",
        "news_limit": 3,
        "retrieve_k": 5,
        "include_analysis": True
    }
    
    result = call_orchestrator("What's the latest on AAPL stock?", params)
    
    assert result is None


def test_call_stt_success(requests_mock, sample_audio_bytes, mock_env):
    """Test call_stt returns parsed JSON when request is successful."""
    mock_response = {"text": "What is the market outlook for tech stocks?"}
    requests_mock.post("http://test-voice:8006/stt", json=mock_response)
    
    result = call_stt(sample_audio_bytes)
    
    assert result == mock_response
    assert "text" in result
    assert result["text"] == "What is the market outlook for tech stocks?"


def test_call_stt_error(requests_mock, sample_audio_bytes, mock_env):
    """Test call_stt returns empty dict with text key when request fails."""
    requests_mock.post(
        "http://test-voice:8006/stt",
        status_code=500,
        json={"error": "Internal server error"}
    )
    
    result = call_stt(sample_audio_bytes)
    
    assert result == {"text": ""}


def test_call_tts_success(requests_mock, mock_env):
    """Test call_tts returns audio bytes when request is successful."""
    mock_audio_content = b"synthesized audio content"
    requests_mock.post(
        "http://test-voice:8006/tts",
        content=mock_audio_content
    )
    
    result = call_tts("Here is your financial brief", {})
    
    assert result == mock_audio_content
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_call_tts_error(requests_mock, mock_env):
    """Test call_tts returns None when request fails."""
    requests_mock.post(
        "http://test-voice:8006/tts",
        status_code=500,
        json={"error": "Internal server error"}
    )
    
    result = call_tts("Here is your financial brief", {})
    
    assert result is None


def test_generate_pdf():
    """Test generate_pdf returns a non-empty bytes stream that contains PDF content."""
    text = "This is a test financial brief with some content."
    filename = "test_brief.pdf"
    
    result = generate_pdf(text, filename)
    
    assert result is not None
    assert isinstance(result, (bytes, bytearray))
    assert len(result) > 0
    # Check if the PDF header is present in the output
    assert b"%PDF" in result
