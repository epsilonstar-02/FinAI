import pytest
from unittest.mock import patch, AsyncMock, MagicMock, Mock
import io
import os
import warnings

from agents.voice_agent.stt_client import STTClient, STTClientError

# Mock the rnnoise module as not available
import sys
sys.modules['rnnoise'] = None


class TestSTTClient:
    """Tests for the Speech-to-Text client"""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Set up common mocks for all tests."""
        # Mock the whisper model
        self.mock_whisper = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.transcribe.return_value = {
            "text": "This is a test transcription.",
            "confidence": 0.95
        }
        self.mock_whisper.load_model.return_value = self.mock_model
        
        # Mock VAD
        self.mock_vad = MagicMock()
        self.mock_vad_instance = MagicMock()
        self.mock_vad_instance.is_speech.return_value = True
        self.mock_vad.return_value = self.mock_vad_instance
        
        # Apply patches
        self.patchers = [
            patch('agents.voice_agent.stt_client.openai_whisper', self.mock_whisper),
            patch('agents.voice_agent.stt_client.webrtcvad.Vad', self.mock_vad),
            patch('agents.voice_agent.stt_client.RNNOISE_AVAILABLE', False),  # RNNoise not available
            # Make sure the whisper module is not imported directly
            patch.dict('sys.modules', {'whisper': None})
        ]
        
        for patcher in self.patchers:
            patcher.start()
        
        # Suppress warnings during tests
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Create the client
        self.client = STTClient()
        
        # Make sure the client was created without RNNoise
        assert self.client.rnnoise is None
        
        yield
        
        # Clean up patches
        for patcher in self.patchers:
            patcher.stop()
    
    @pytest.fixture
    def stt_client(self):
        """Return the STT client instance with mocks applied."""
        return self.client
        
    def test_rnnoise_not_available(self, stt_client):
        """Test that the client works when RNNoise is not available."""
        # The client should still be usable even without RNNoise
        assert stt_client is not None
        assert stt_client.rnnoise is None

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing"""
        return b"mock audio bytes"

    @patch("agents.voice_agent.stt_client.AudioSegment")
    async def test_transcribe_whisper_success(self, mock_audio_segment, stt_client, sample_audio):
        """Test successful transcription with Whisper"""
        # Mock AudioSegment for processing
        mock_segment = MagicMock()
        mock_audio_segment.from_file.return_value = mock_segment
        mock_segment.raw_data = b"processed audio data"
        mock_segment.set_channels.return_value = mock_segment
        mock_segment.set_frame_rate.return_value = mock_segment
        mock_segment.set_sample_width.return_value = mock_segment
        
        # Call the method
        text, confidence = await stt_client.transcribe_whisper(sample_audio)
        
        # Verify results
        assert text == "This is a test transcription."
        assert confidence == 0.95
        
        # Verify model was used
        assert stt_client.model.transcribe.called

    async def test_transcribe_whisper_error(self, stt_client, sample_audio):
        """Test error handling in Whisper transcription"""
        # Force an error
        stt_client.model = MagicMock()
        stt_client.model.transcribe.side_effect = Exception("Test transcription error")
        
        # Verify exception is raised and wrapped properly
        with pytest.raises(STTClientError) as excinfo:
            await stt_client.transcribe_whisper(sample_audio)
        
        assert "Failed to transcribe audio" in str(excinfo.value)

    @patch("agents.voice_agent.stt_client.AudioSegment")
    @patch("agents.voice_agent.stt_client.settings")
    async def test_denoise_audio(self, mock_settings, mock_audio_segment, stt_client, sample_audio):
        """Test audio denoising functionality"""
        # Test when RNNoise is disabled
        mock_settings.RNNOISE_ENABLED = False
        result = stt_client._denoise_audio(sample_audio)
        assert result == sample_audio  # Should return original audio when disabled
        
        # Test when RNNoise is enabled but not available
        mock_settings.RNNOISE_ENABLED = True
        stt_client.rnnoise = None
        result = stt_client._denoise_audio(sample_audio)
        assert result == sample_audio  # Should return original audio when not available
        
        # Test when RNNoise is enabled and available
        mock_rnnoise = MagicMock()
        mock_rnnoise.process_frame.return_value = b"denoised_audio"
        stt_client.rnnoise = mock_rnnoise
        
        # Mock AudioSegment for processing
        mock_segment = MagicMock()
        mock_segment.raw_data = b"raw_audio_data"
        mock_segment.set_channels.return_value = mock_segment
        mock_segment.set_frame_rate.return_value = mock_segment
        mock_segment.set_sample_width.return_value = mock_segment
        mock_audio_segment.from_file.return_value = mock_segment
        
        # Call the method
        result = stt_client._denoise_audio(sample_audio)
        
        # Verify denoising was applied
        assert result != sample_audio  # Should return processed audio
        mock_rnnoise.process_frame.assert_called()

    @patch("agents.voice_agent.stt_client.AudioSegment")
    async def test_chunk_audio_with_vad(self, mock_audio_segment, stt_client, sample_audio):
        """Test audio chunking with VAD"""
        # Mock AudioSegment for processing
        mock_segment = MagicMock()
        mock_audio_segment.from_file.return_value = mock_segment
        mock_segment.raw_data = b"raw audio data" * 1000  # Make it long enough for multiple chunks
        mock_segment.set_channels.return_value = mock_segment
        mock_segment.set_frame_rate.return_value = mock_segment
        mock_segment.set_sample_width.return_value = mock_segment
        
        # Call the method
        chunks = stt_client._chunk_audio_with_vad(sample_audio)
        
        # Verify chunks were created
        assert len(chunks) > 0

    async def test_transcribe_provider_selection(self, stt_client, sample_audio):
        """Test provider selection logic"""
        # Mock both provider methods
        stt_client.transcribe_whisper = AsyncMock(return_value=("Whisper result", 0.9))
        stt_client.google_stt = AsyncMock(return_value=("Google result", 0.8))
        
        # Test Whisper provider
        with patch("agents.voice_agent.stt_client.settings.STT_PROVIDER", "whisper"):
            result = await stt_client.transcribe(sample_audio)
            assert result[0] == "Whisper result"
            assert stt_client.transcribe_whisper.called
            assert not stt_client.google_stt.called
        
        # Reset mocks
        stt_client.transcribe_whisper.reset_mock()
        stt_client.google_stt.reset_mock()
        
        # Test Google provider
        with patch("agents.voice_agent.stt_client.settings.STT_PROVIDER", "google-stt"):
            result = await stt_client.transcribe(sample_audio)
            assert result[0] == "Google result"
            assert stt_client.google_stt.called
            assert not stt_client.transcribe_whisper.called

    async def test_transcribe_invalid_provider(self, stt_client, sample_audio):
        """Test error handling for invalid provider"""
        with patch("agents.voice_agent.stt_client.settings.STT_PROVIDER", "invalid-provider"):
            with pytest.raises(STTClientError) as excinfo:
                await stt_client.transcribe(sample_audio)
            
            assert "Unsupported STT provider" in str(excinfo.value)
