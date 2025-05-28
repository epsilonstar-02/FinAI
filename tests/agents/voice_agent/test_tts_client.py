import pytest
from unittest.mock import patch, AsyncMock, MagicMock, Mock
import io
import os
import hashlib

from agents.voice_agent.tts_client import TTSClient, TTSClientError


class TestTTSClient:
    """Tests for the Text-to-Speech client"""

    @pytest.fixture
    def tts_client(self):
        """Create a TTS client instance for testing"""
        with patch("agents.voice_agent.tts_client.diskcache.Cache") as mock_cache:
            # Mock the cache
            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            
            # Mock gTTS
            with patch("agents.voice_agent.tts_client.gTTS") as mock_gtts:
                mock_gtts_instance = MagicMock()
                mock_gtts.return_value = mock_gtts_instance
                
                # Mock Gemini
                with patch("agents.voice_agent.tts_client.genai") as mock_genai:
                    client = TTSClient()
                    # Store the mock gTTS class on the client for later use in tests
                    client._TTSClient__gtts = mock_gtts
                    yield client

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return "This is a test text for speech synthesis."

    @patch("agents.voice_agent.tts_client.AudioSegment")
    async def test_synthesize_gtts_success(self, mock_audio_segment, tts_client, sample_text):
        """Test successful speech synthesis with gTTS"""
        # Create a real BytesIO object to capture the output
        mock_mp3_data = b"mock mp3 data"
        
        # Set up the cache to return None (cache miss)
        tts_client.cache.get.return_value = None
        
        # Mock the gTTS instance and its write_to_fp method
        mock_gtts_instance = MagicMock()
        
        def write_to_fp(fp):
            fp.write(mock_mp3_data)
            
        mock_gtts_instance.write_to_fp = write_to_fp
        
        # Set up the gTTS mock to return our mock instance
        tts_client._TTSClient__gtts.return_value = mock_gtts_instance
        
        # Call the method
        result = await tts_client.synthesize_gtts(sample_text)
        
        # Verify results
        assert result == mock_mp3_data
        
        # Verify gTTS was called with the correct arguments
        tts_client._TTSClient__gtts.assert_called_once_with(
            text=sample_text, 
            lang="en", 
            slow=False
        )
        
        # Verify cache was checked
        assert tts_client.cache.get.called
        
        # Verify cache was updated with the correct value
        cache_key = tts_client._get_cache_key(sample_text, "default", 1.0)
        tts_client.cache.__setitem__.assert_called_once_with(cache_key, mock_mp3_data)

    @patch("agents.voice_agent.tts_client.AudioSegment")
    async def test_synthesize_gtts_error(self, mock_audio_segment, tts_client, sample_text):
        """Test error handling in gTTS synthesis"""
        # Set up the cache to return None (cache miss)
        tts_client.cache.get.return_value = None
        
        # Mock the gTTS class to raise an exception
        mock_gtts_instance = MagicMock()
        mock_gtts_instance.write_to_fp.side_effect = Exception("Test synthesis error")
        tts_client._TTSClient__gtts.return_value = mock_gtts_instance
        
        # Verify exception is raised and wrapped properly
        with pytest.raises(TTSClientError) as excinfo:
            await tts_client.synthesize_gtts(sample_text)
        
        # Verify the error message is properly wrapped
        assert "Failed to synthesize speech with gTTS" in str(excinfo.value)
        assert "Test synthesis error" in str(excinfo.value)
        
        # Verify cache was checked but not updated
        assert tts_client.cache.get.called
        assert not tts_client.cache.__setitem__.called

    @patch("agents.voice_agent.tts_client.AudioSegment")
    async def test_synthesize_gtts_with_speed(self, mock_audio_segment, tts_client, sample_text):
        """Test speech synthesis with speed adjustment"""
        # Set up the cache to return None (cache miss)
        tts_client.cache.get.return_value = None
        
        # Mock the gTTS instance and its write_to_fp method
        mock_gtts_instance = MagicMock()
        mock_mp3_data = b"mock mp3 data"
        
        def write_to_fp(fp):
            fp.write(mock_mp3_data)
            
        mock_gtts_instance.write_to_fp = write_to_fp
        tts_client._TTSClient__gtts.return_value = mock_gtts_instance
        
        # Mock AudioSegment for speed adjustment
        mock_segment = MagicMock()
        mock_audio_segment.from_file.return_value = mock_segment
        
        # Create a mock for the spawned segment (after speed adjustment)
        mock_spawned_segment = MagicMock()
        mock_segment._spawn.return_value = mock_spawned_segment
        
        # Mock the export method to return our test data
        mock_export = MagicMock()
        mock_spawned_segment.export = mock_export
        
        # Call the method with speed adjustment
        result = await tts_client.synthesize_gtts(sample_text, speed=1.5)
        
        # Verify gTTS was called with the correct arguments
        tts_client._TTSClient__gtts.assert_called_once_with(
            text=sample_text, 
            lang="en", 
            slow=False
        )
        
        # Verify AudioSegment.from_file was called with the correct arguments
        mock_audio_segment.from_file.assert_called_once()
        
        # Verify speed adjustment was applied
        mock_segment._spawn.assert_called_once()
        
        # Verify export was called
        mock_export.assert_called_once()
        
        # Verify cache was updated with the correct value
        cache_key = tts_client._get_cache_key(sample_text, "default", 1.5)
        tts_client.cache.__setitem__.assert_called_once()

    async def test_get_cache_key(self, tts_client, sample_text):
        """Test cache key generation"""
        # Generate cache key
        key = tts_client._get_cache_key(sample_text, "default", 1.0)
        
        # Verify it's a valid MD5 hash
        assert len(key) == 32
        
        # Verify different inputs produce different keys
        key2 = tts_client._get_cache_key(sample_text, "female", 1.0)
        assert key != key2

    async def test_synthesize_cached_result(self, tts_client, sample_text):
        """Test retrieving cached synthesis result"""
        # Mock cache hit
        cached_data = b"cached audio data"
        tts_client.cache.get.return_value = cached_data
        
        # Call the method
        result = await tts_client.synthesize_gtts(sample_text)
        
        # Verify cached result was returned
        assert result == cached_data
        assert tts_client.cache.get.called
        # Verify no synthesis was performed
        assert not tts_client.cache.__setitem__.called

    @patch("agents.voice_agent.tts_client.genai.GenerativeModel")
    @patch("agents.voice_agent.tts_client.AudioSegment")
    async def test_gemini_tts(self, mock_audio_segment, mock_genai_model, tts_client, sample_text):
        """Test Gemini TTS synthesis"""
        # Set up the cache to return None (cache miss)
        tts_client.cache.get.return_value = None
        
        # Mock Gemini model and response
        mock_model = MagicMock()
        mock_genai_model.return_value = mock_model
        
        # Create a mock response that matches the expected structure
        mock_audio = MagicMock()
        mock_audio.bytes = b"gemini audio data"
        
        mock_part = MagicMock()
        mock_part.audio = mock_audio
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        # Make generate_content return our mock response
        mock_model.generate_content.return_value = mock_response
        
        # Mock AudioSegment to return a dummy segment
        mock_audio_segment_instance = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio_segment_instance
        mock_audio_segment_instance._spawn.return_value = mock_audio_segment_instance
        
        # Call the method
        with patch("agents.voice_agent.tts_client.io.BytesIO"):
            result = await tts_client.gemini_tts(sample_text)
            
            # Verify the result is the mock audio data
            assert result == b"gemini audio data"
            
            # Verify the model was called with the correct arguments
            mock_genai_model.assert_called_once_with('gemini-tts')
            mock_model.generate_content.assert_called_once_with(
                sample_text,
                generation_config={"voice": None},
                stream=False
            )
            
            # Verify cache was checked and updated
            assert tts_client.cache.get.called
            cache_key = tts_client._get_cache_key(sample_text, "default", 1.0)
            tts_client.cache.__setitem__.assert_called_once_with(cache_key, b"gemini audio data")

    async def test_synthesize_provider_selection(self, tts_client, sample_text):
        """Test provider selection logic"""
        # Mock both provider methods
        tts_client.synthesize_gtts = AsyncMock(return_value=b"gTTS result")
        tts_client.gemini_tts = AsyncMock(return_value=b"Gemini result")
        
        # Test gTTS provider
        with patch("agents.voice_agent.tts_client.settings.TTS_PROVIDER", "gtts"):
            result = await tts_client.synthesize(sample_text)
            assert result == b"gTTS result"
            assert tts_client.synthesize_gtts.called
            assert not tts_client.gemini_tts.called
        
        # Reset mocks
        tts_client.synthesize_gtts.reset_mock()
        tts_client.gemini_tts.reset_mock()
        
        # Test Gemini provider
        with patch("agents.voice_agent.tts_client.settings.TTS_PROVIDER", "gemini-tts"):
            result = await tts_client.synthesize(sample_text)
            assert result == b"Gemini result"
            assert tts_client.gemini_tts.called
            assert not tts_client.synthesize_gtts.called

    async def test_synthesize_invalid_provider(self, tts_client, sample_text):
        """Test error handling for invalid provider"""
        with patch("agents.voice_agent.tts_client.settings.TTS_PROVIDER", "invalid-provider"):
            with pytest.raises(TTSClientError) as excinfo:
                await tts_client.synthesize(sample_text)
            
            assert "Unsupported TTS provider" in str(excinfo.value)
