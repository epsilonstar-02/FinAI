"""
Multi-provider Text-to-Speech client that supports multiple TTS services with fallback mechanisms.
"""
import os
import io
import logging
import asyncio
import time
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from pathlib import Path
import traceback
from functools import wraps

# Caching
from diskcache import Cache

# Retry utilities
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

# Audio processing
import numpy as np
from pydub import AudioSegment

# TTS providers
# pyttsx3 (offline)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# gTTS (Google Text-to-Speech)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Microsoft Edge TTS
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# ElevenLabs
try:
    from elevenlabs import generate, set_api_key
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

# AWS Polly
try:
    import boto3
    AWS_POLLY_AVAILABLE = True
except ImportError:
    AWS_POLLY_AVAILABLE = False

# Silero TTS
try:
    import torch
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

# Coqui TTS
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

# Local imports
from .config import settings, TTSProvider

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Base exception for TTS errors."""
    pass


class ProviderError(TTSError):
    """Exception for specific provider errors."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"{provider} error: {message}")


class AllProvidersFailedError(TTSError):
    """Exception when all providers fail."""
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors
        error_msg = "; ".join([f"{p}: {e}" for p, e in provider_errors.items()])
        super().__init__(f"All TTS providers failed: {error_msg}")


class MultiTTSClient:
    """Client that supports multiple TTS providers with fallback mechanisms."""
    
    def __init__(self):
        """Initialize the multi-TTS client."""
        self.available_providers = self._initialize_providers()
        self.provider_errors = {}
        
        # Set up caching if enabled
        if settings.ENABLE_CACHE:
            self.cache = Cache(settings.CACHE_DIR, size_limit=1e9)  # 1GB limit
        else:
            self.cache = None
        
        if not self.available_providers:
            logger.warning("No TTS providers are available. Check configuration.")
    
    def _initialize_providers(self) -> List[TTSProvider]:
        """Initialize the available TTS providers."""
        available = []
        
        # Check each provider
        for provider in TTSProvider:
            if settings.is_tts_provider_available(provider):
                try:
                    # Perform provider-specific initialization
                    if provider == TTSProvider.PYTTSX3 and PYTTSX3_AVAILABLE:
                        # Initialize pyttsx3 engine
                        if not hasattr(self, '_pyttsx3_engine'):
                            self._pyttsx3_engine = None  # Lazy loading
                        available.append(provider)
                    
                    elif provider == TTSProvider.GTTS and GTTS_AVAILABLE:
                        available.append(provider)
                    
                    elif provider == TTSProvider.EDGE and EDGE_TTS_AVAILABLE:
                        available.append(provider)
                    
                    elif provider == TTSProvider.ELEVENLABS and ELEVENLABS_AVAILABLE:
                        if settings.ELEVENLABS_API_KEY:
                            set_api_key(settings.ELEVENLABS_API_KEY)
                            available.append(provider)
                    
                    elif provider == TTSProvider.AMAZON_POLLY and AWS_POLLY_AVAILABLE:
                        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                            # Initialize boto3 client
                            if not hasattr(self, '_polly_client'):
                                self._polly_client = None  # Lazy loading
                            available.append(provider)
                    
                    elif provider == TTSProvider.SILERO and SILERO_AVAILABLE:
                        # Initialize Silero model
                        if not hasattr(self, '_silero_model'):
                            self._silero_model = None  # Lazy loading
                        available.append(provider)
                    
                    elif provider == TTSProvider.COQUI and COQUI_AVAILABLE:
                        # Initialize Coqui model
                        if not hasattr(self, '_coqui_tts'):
                            self._coqui_tts = None  # Lazy loading
                        available.append(provider)
                
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider}: {str(e)}")
        
        logger.info(f"Available TTS providers: {[p.value for p in available]}")
        return available
    
    def _get_cache_key(self, text: str, voice: str) -> str:
        """Generate a cache key for the TTS request."""
        import hashlib
        # Create a hash of the text and voice
        return hashlib.md5(f"{text}:{voice}".encode()).hexdigest()
    
    async def _synthesize_pyttsx3(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using pyttsx3 (offline)."""
        if not PYTTSX3_AVAILABLE:
            raise ProviderError("pyttsx3", "pyttsx3 is not installed")
        
        try:
            # Lazy load the engine
            if not hasattr(self, '_pyttsx3_engine') or self._pyttsx3_engine is None:
                logger.info("Initializing pyttsx3 engine...")
                self._pyttsx3_engine = pyttsx3.init()
            
            # Set voice if specified
            if voice:
                voices = self._pyttsx3_engine.getProperty('voices')
                voice_found = False
                for v in voices:
                    if voice.lower() in v.id.lower() or voice.lower() in v.name.lower():
                        self._pyttsx3_engine.setProperty('voice', v.id)
                        voice_found = True
                        break
                
                if not voice_found:
                    logger.warning(f"Voice '{voice}' not found, using default voice")
            
            # Set speech rate
            rate = self._pyttsx3_engine.getProperty('rate')
            self._pyttsx3_engine.setProperty('rate', int(rate / speed))
            
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save to file
            self._pyttsx3_engine.save_to_file(text, temp_path)
            
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._pyttsx3_engine.runAndWait)
            
            # Read the file and convert to MP3
            with open(temp_path, "rb") as f:
                wav_data = f.read()
            
            # Convert WAV to MP3 using pydub
            audio = AudioSegment.from_wav(io.BytesIO(wav_data))
            mp3_io = io.BytesIO()
            audio.export(mp3_io, format="mp3")
            mp3_data = mp3_io.getvalue()
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return mp3_data
        
        except Exception as e:
            raise ProviderError("pyttsx3", str(e))
    
    async def _synthesize_gtts(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using Google Text-to-Speech."""
        if not GTTS_AVAILABLE:
            raise ProviderError("gTTS", "gTTS is not installed")
        
        try:
            # gTTS doesn't support voice selection, only language
            lang = "en"
            if voice:
                # Extract language code from voice if possible
                if len(voice) == 2:
                    lang = voice
                elif "-" in voice:
                    lang = voice.split("-")[0]
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=speed < 1.0)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: tts.save(temp_path))
            
            # Read the file
            with open(temp_path, "rb") as f:
                mp3_data = f.read()
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Apply speed adjustment if needed (other than slow/normal)
            if speed != 1.0 and not (speed < 1.0):
                # Load audio and adjust speed
                audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                # Speed up by using segment overlay technique
                if speed > 1.0:
                    audio = audio.speedup(playback_speed=speed)
                # Export back to MP3
                mp3_io = io.BytesIO()
                audio.export(mp3_io, format="mp3")
                mp3_data = mp3_io.getvalue()
            
            return mp3_data
        
        except Exception as e:
            raise ProviderError("gTTS", str(e))
    
    async def _synthesize_edge_tts(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using Microsoft Edge TTS."""
        if not EDGE_TTS_AVAILABLE:
            raise ProviderError("Edge TTS", "edge-tts is not installed")
        
        try:
            # Default voice if not specified
            if not voice:
                voice = "en-US-AriaNeural"
            
            # Create output buffer
            output = io.BytesIO()
            
            # Format speed as a string percent
            speed_str = f"{int(speed * 100)}%"
            
            # Communicate with Edge TTS
            communicate = edge_tts.Communicate(text, voice, rate=speed_str)
            
            # Run in an async context
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    output.write(chunk["data"])
            
            # Return the audio data
            output.seek(0)
            mp3_data = output.read()
            
            return mp3_data
        
        except Exception as e:
            raise ProviderError("Edge TTS", str(e))
    
    async def _synthesize_elevenlabs(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using ElevenLabs."""
        if not ELEVENLABS_AVAILABLE:
            raise ProviderError("ElevenLabs", "ElevenLabs is not installed")
        
        try:
            # Use specified voice or default from settings
            voice_id = voice or settings.ELEVENLABS_VOICE_ID
            if not voice_id:
                raise ProviderError("ElevenLabs", "No voice ID specified")
            
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                lambda: generate(
                    text=text,
                    voice=voice_id,
                    model="eleven_monolingual_v1"
                )
            )
            
            # ElevenLabs returns MP3 data directly
            return audio_data
        
        except Exception as e:
            raise ProviderError("ElevenLabs", str(e))
    
    async def _synthesize_amazon_polly(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using Amazon Polly."""
        if not AWS_POLLY_AVAILABLE:
            raise ProviderError("Amazon Polly", "boto3 is not installed")
        
        try:
            # Lazy load the client
            if not hasattr(self, '_polly_client') or self._polly_client is None:
                logger.info("Initializing Amazon Polly client...")
                self._polly_client = boto3.client(
                    'polly',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
            
            # Use specified voice or default from settings
            voice_id = voice or settings.POLLY_VOICE_ID
            
            # Add SSML for speed adjustment if needed
            if speed != 1.0:
                text = f'<speak><prosody rate="{int(speed * 100)}%">{text}</prosody></speak>'
                text_type = "ssml"
            else:
                text_type = "text"
            
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._polly_client.synthesize_speech(
                    Text=text,
                    TextType=text_type,
                    OutputFormat="mp3",
                    VoiceId=voice_id,
                    Engine="neural"
                )
            )
            
            # Extract audio data
            if "AudioStream" in response:
                audio_stream = response["AudioStream"]
                audio_data = audio_stream.read()
                return audio_data
            else:
                raise ProviderError("Amazon Polly", "No audio data in response")
        
        except Exception as e:
            raise ProviderError("Amazon Polly", str(e))
    
    async def _synthesize_silero(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using Silero TTS (offline)."""
        if not SILERO_AVAILABLE:
            raise ProviderError("Silero", "PyTorch is not installed")
        
        try:
            # Lazy load the model
            if not hasattr(self, '_silero_model') or self._silero_model is None:
                logger.info("Loading Silero TTS model...")
                
                # Check if custom model path exists
                if settings.SILERO_MODEL_PATH and Path(settings.SILERO_MODEL_PATH).exists():
                    model_path = settings.SILERO_MODEL_PATH
                    custom_model = True
                else:
                    # Download model from torch hub
                    model_path = "silero_tts"
                    custom_model = False
                
                # Load the model
                if custom_model:
                    self._silero_model = torch.jit.load(model_path)
                else:
                    self._silero_model, _ = torch.hub.load(
                        repo_or_dir="snakers4/silero-models",
                        model="silero_tts",
                        language=settings.SILERO_LANGUAGE,
                        speaker=settings.SILERO_SPEAKER
                    )
            
            # Use specified voice or default from settings
            speaker = voice or settings.SILERO_SPEAKER
            
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._silero_model.save_wav(
                    text=text,
                    speaker=speaker,
                    sample_rate=settings.SAMPLE_RATE,
                    audio_path=temp_path
                )
            )
            
            # Read the file and convert to MP3
            with open(temp_path, "rb") as f:
                wav_data = f.read()
            
            # Convert WAV to MP3 using pydub
            audio = AudioSegment.from_wav(io.BytesIO(wav_data))
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Speed up or slow down
                if speed > 1.0:
                    audio = audio.speedup(playback_speed=speed)
                else:
                    # This is a simple way to slow down, not ideal
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * speed)
                    }).set_frame_rate(audio.frame_rate)
            
            # Export to MP3
            mp3_io = io.BytesIO()
            audio.export(mp3_io, format="mp3")
            mp3_data = mp3_io.getvalue()
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return mp3_data
        
        except Exception as e:
            raise ProviderError("Silero", str(e))
    
    async def _synthesize_coqui(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize speech using Coqui TTS (offline)."""
        if not COQUI_AVAILABLE:
            raise ProviderError("Coqui", "Coqui TTS is not installed")
        
        try:
            # Lazy load the model
            if not hasattr(self, '_coqui_tts') or self._coqui_tts is None:
                logger.info("Loading Coqui TTS model...")
                
                # Check if custom model path exists
                if settings.COQUI_MODEL_PATH and Path(settings.COQUI_MODEL_PATH).exists():
                    model_path = settings.COQUI_MODEL_PATH
                    self._coqui_tts = CoquiTTS(model_path=model_path)
                else:
                    # Use built-in model
                    self._coqui_tts = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._coqui_tts.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    speaker=voice if voice else None
                )
            )
            
            # Read the file and convert to MP3
            with open(temp_path, "rb") as f:
                wav_data = f.read()
            
            # Convert WAV to MP3 using pydub
            audio = AudioSegment.from_wav(io.BytesIO(wav_data))
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Speed up or slow down
                if speed > 1.0:
                    audio = audio.speedup(playback_speed=speed)
                else:
                    # This is a simple way to slow down, not ideal
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * speed)
                    }).set_frame_rate(audio.frame_rate)
            
            # Export to MP3
            mp3_io = io.BytesIO()
            audio.export(mp3_io, format="mp3")
            mp3_data = mp3_io.getvalue()
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return mp3_data
        
        except Exception as e:
            raise ProviderError("Coqui", str(e))
    
    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=settings.RETRY_DELAY, max=10),
        reraise=True
    )
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0, provider: Optional[TTSProvider] = None) -> bytes:
        """
        Synthesize speech from text using the specified or default provider.
        Falls back to other providers if the specified provider fails.
        
        Args:
            text: Text to synthesize
            voice: Optional voice name or ID
            speed: Speed factor (1.0 is normal speed)
            provider: Optional specific provider to use
            
        Returns:
            Audio data as bytes (MP3 format)
            
        Raises:
            TTSError: If all providers fail
        """
        # Reset provider errors
        self.provider_errors = {}
        
        # Check the cache first if enabled
        if settings.ENABLE_CACHE and self.cache:
            cache_key = self._get_cache_key(text, voice or "default")
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Using cached TTS result")
                return cached_result
        
        # Determine which providers to try
        providers_to_try = []
        
        if provider and provider in self.available_providers:
            # If a specific provider is requested and available, try it first
            providers_to_try.append(provider)
        
        # Then add the default provider if it's not already in the list
        if settings.DEFAULT_TTS_PROVIDER in self.available_providers and settings.DEFAULT_TTS_PROVIDER not in providers_to_try:
            providers_to_try.append(settings.DEFAULT_TTS_PROVIDER)
        
        # Then add fallback providers if they're not already in the list
        for fallback in settings.FALLBACK_TTS_PROVIDERS:
            if fallback in self.available_providers and fallback not in providers_to_try:
                providers_to_try.append(fallback)
        
        # Finally, add any other available providers not already in the list
        for available in self.available_providers:
            if available not in providers_to_try:
                providers_to_try.append(available)
        
        # If no providers are available, raise an error
        if not providers_to_try:
            raise TTSError("No TTS providers are available. Check configuration.")
        
        # Try each provider in order
        for provider in providers_to_try:
            try:
                if provider == TTSProvider.PYTTSX3:
                    audio_data = await self._synthesize_pyttsx3(text, voice, speed)
                elif provider == TTSProvider.GTTS:
                    audio_data = await self._synthesize_gtts(text, voice, speed)
                elif provider == TTSProvider.EDGE:
                    audio_data = await self._synthesize_edge_tts(text, voice, speed)
                elif provider == TTSProvider.ELEVENLABS:
                    audio_data = await self._synthesize_elevenlabs(text, voice, speed)
                elif provider == TTSProvider.AMAZON_POLLY:
                    audio_data = await self._synthesize_amazon_polly(text, voice, speed)
                elif provider == TTSProvider.SILERO:
                    audio_data = await self._synthesize_silero(text, voice, speed)
                elif provider == TTSProvider.COQUI:
                    audio_data = await self._synthesize_coqui(text, voice, speed)
                else:
                    # Skip unsupported providers
                    logger.warning(f"Unsupported provider: {provider}")
                    continue
                
                # If we got a result, cache it and return
                if audio_data:
                    # Cache the result if enabled
                    if settings.ENABLE_CACHE and self.cache:
                        cache_key = self._get_cache_key(text, voice or "default")
                        self.cache.set(cache_key, audio_data, expire=settings.CACHE_TTL)
                    
                    return audio_data
                else:
                    # If no audio was returned, log and try the next provider
                    logger.warning(f"Provider {provider} returned no audio")
                    self.provider_errors[provider.value] = "No audio returned"
                    continue
            
            except ProviderError as e:
                # Log the error and continue to the next provider
                logger.warning(f"Provider {provider} failed: {str(e)}")
                self.provider_errors[provider.value] = str(e)
                continue
        
        # If we've tried all providers and all have failed, raise an error
        raise AllProvidersFailedError(self.provider_errors)
    
    def get_provider_errors(self) -> Dict[str, str]:
        """Get the errors that occurred for each provider."""
        return self.provider_errors


# Create a singleton instance
_multi_tts_client = None

def get_multi_tts_client() -> MultiTTSClient:
    """Get the multi-TTS client singleton instance."""
    global _multi_tts_client
    if _multi_tts_client is None:
        _multi_tts_client = MultiTTSClient()
    return _multi_tts_client


async def synthesize(text: str, voice: Optional[str] = None, speed: float = 1.0, provider: Optional[TTSProvider] = None) -> bytes:
    """
    Synthesize speech from text using the multi-TTS client.
    
    Args:
        text: Text to synthesize
        voice: Optional voice name or ID
        speed: Speed factor (1.0 is normal speed)
        provider: Optional specific provider to use
        
    Returns:
        Audio data as bytes (MP3 format)
        
    Raises:
        TTSError: If an error occurs in synthesis
    """
    client = get_multi_tts_client()
    return await client.synthesize(text, voice, speed, provider)
