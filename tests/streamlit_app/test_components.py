import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Update path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from streamlit_app.components import render_card, render_audio_player, show_progress_step


@pytest.fixture
def sample_audio_bytes():
    """Create a small sample of audio bytes for testing."""
    return b"dummy audio bytes for testing"


@pytest.fixture
def streamlit_mock():
    """Create a mock for streamlit functions."""
    with patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.audio") as mock_audio, \
         patch("streamlit.spinner") as mock_spinner, \
         patch("streamlit.success") as mock_success:
        
        # Configure mocks to return a DeltaGenerator-like object
        mock_delta_gen = MagicMock()
        mock_markdown.return_value = mock_delta_gen
        mock_audio.return_value = mock_delta_gen
        mock_spinner.return_value.__enter__.return_value = None
        mock_spinner.return_value.__exit__.return_value = None
        mock_success.return_value = mock_delta_gen
        
        yield {
            "markdown": mock_markdown,
            "audio": mock_audio,
            "spinner": mock_spinner,
            "success": mock_success,
            "delta_gen": mock_delta_gen
        }


def test_render_card(streamlit_mock):
    """Test that render_card runs without error and returns a Streamlit DeltaGenerator."""
    title = "Test Title"
    content = "Test Content"
    
    result = render_card(title, content)
    
    # Verify that st.markdown was called with HTML content
    streamlit_mock["markdown"].assert_called_once()
    call_args = streamlit_mock["markdown"].call_args[0][0]
    
    # Check that the title and content are in the HTML
    assert title in call_args
    assert content in call_args
    assert "card-title" in call_args
    assert "card-content" in call_args
    
    # Verify that unsafe_allow_html is set to True
    assert streamlit_mock["markdown"].call_args[1]["unsafe_allow_html"] is True


def test_render_audio_player(streamlit_mock, sample_audio_bytes):
    """Test that render_audio_player runs without error given a small bytes fixture."""
    result = render_audio_player(sample_audio_bytes)
    
    # Verify that st.audio was called with the audio bytes
    streamlit_mock["audio"].assert_called_once_with(sample_audio_bytes, format="audio/mp3")


@patch("time.sleep")
@patch("time.time")
def test_show_progress_step(mock_time, mock_sleep, streamlit_mock):
    """Test that show_progress_step runs without error and shows spinner and success."""
    # Configure time.time to return different values on consecutive calls
    mock_time.side_effect = [10.0, 11.5]  # Start time, end time (1.5s elapsed)
    
    show_progress_step("Test Step", 1000)
    
    # Verify that spinner was called with the correct message
    streamlit_mock["spinner"].assert_called_once_with("Test Step...")
    
    # Verify that sleep was called with the correct duration
    mock_sleep.assert_called_once_with(1.0)  # 1000ms = 1.0s
    
    # Verify that success was called with the correct message
    streamlit_mock["success"].assert_called_once()
    success_msg = streamlit_mock["success"].call_args[0][0]
    assert "Test Step completed in 1.50s" in success_msg
