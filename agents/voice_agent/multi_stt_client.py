"""
Multi-provider Speech-to-Text client that supports multiple STT services with fallback mechanisms.
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
import webrtcvad
import wave

# STT providers
import speech_recognition as sr
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from google.cloud import speech
    GOOGLE_CLOUD_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_SPEECH_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as azure_speech
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import deepspeech
    DEEPSPEECH_AVAILABLE = True
except ImportError:
    DEEPSPEECH_AVAILABLE = False

# Local imports
from .config import settings, STTProvider

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class STTError(Exception):
    """Base exception for STT errors."""
    pass


class ProviderError(STTError):
    """Exception for specific provider errors."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"{provider} error: {message}")


class AllProvidersFailedError(STTError):
    """Exception when all providers fail."""
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors
        error_msg = "; ".join([f"{p}: {e}" for p, e in provider_errors.items()])
        super().__init__(f"All STT providers failed: {error_msg}")


class MultiSTTClient:
    """Client that supports multiple STT providers with fallback mechanisms."""
    
    def __init__(self):
        """Initialize the multi-STT client."""
        self.available_providers = self._initialize_providers()
        self.provider_errors = {}
        
        # Set up caching if enabled
        if settings.ENABLE_CACHE:
            self.cache = Cache(settings.CACHE_DIR, size_limit=1e9)  # 1GB limit
        else:
            self.cache = None
        
        # Set up VAD if enabled
        if settings.VAD_ENABLED:
            self.vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
        else:
            self.vad = None
        
        if not self.available_providers:
            logger.warning("No STT providers are available. Check configuration.")
    
    def _initialize_providers(self) -> List[STTProvider]:
        """Initialize the available STT providers."""
        available = []
        
        # Check each provider
        for provider in STTProvider:
            if settings.is_stt_provider_available(provider):
                try:
                    # Perform provider-specific initialization
                    if provider == STTProvider.WHISPER_LOCAL and WHISPER_AVAILABLE:
                        # Initialize Whisper model if it's not loaded already
                        if not hasattr(self, '_whisper_model'):
                            logger.info(f"Loading Whisper model ({settings.WHISPER_MODEL_SIZE})...")
                            self._whisper_model = None  # Lazy loading
                        available.append(provider)
                    
                    elif provider == STTProvider.GOOGLE and GOOGLE_CLOUD_SPEECH_AVAILABLE:
                        available.append(provider)
                    
                    elif provider == STTProvider.AZURE and AZURE_SPEECH_AVAILABLE:
                        available.append(provider)
                    
                    elif provider == STTProvider.VOSK and VOSK_AVAILABLE:
                        if not hasattr(self, '_vosk_model'):
                            self._vosk_model = None  # Lazy loading
                        available.append(provider)
                    
                    elif provider == STTProvider.DEEPSPEECH and DEEPSPEECH_AVAILABLE:
                        if not hasattr(self, '_deepspeech_model'):
                            self._deepspeech_model = None  # Lazy loading
                        available.append(provider)
                    
                    elif provider == STTProvider.SPEECHRECOGNITION:
                        # SpeechRecognition is always available
                        available.append(provider)
                    
                    elif provider == STTProvider.WHISPER_API:
                        # Whisper API is available if API key is set
                        if settings.WHISPER_API_KEY:
                            available.append(provider)
                
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider}: {str(e)}")
        
        logger.info(f"Available STT providers: {[p.value for p in available]}")
        return available
    
    def _preprocess_audio(self, audio_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Preprocess audio data for better transcription results.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Tuple of processed audio data and metadata
        """
        metadata = {}
        
        try:
            # Convert bytes to AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Record original format
            metadata["original_format"] = audio.channels, audio.frame_rate, audio.sample_width
            
            # Resample to 16kHz mono for STT
            if audio.channels != 1 or audio.frame_rate != settings.SAMPLE_RATE:
                audio = audio.set_channels(1).set_frame_rate(settings.SAMPLE_RATE)
            
            # Normalize volume
            audio = audio.normalize()
            
            # Apply noise reduction if enabled
            if settings.NOISE_REDUCTION_ENABLED:
                # Simple noise reduction using high-pass filter
                audio = audio.high_pass_filter(80)
            
            # Convert back to bytes
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            processed_audio = buffer.getvalue()
            
            # Calculate audio duration
            metadata["duration"] = len(audio) / 1000.0  # in seconds
            
            return processed_audio, metadata
        
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {str(e)}")
            # Return original audio if preprocessing fails
            return audio_data, metadata
    
    def _get_cache_key(self, audio_data: bytes) -> str:
        """Generate a cache key for the audio data."""
        import hashlib
        # Use first 100KB of audio for the hash to avoid hashing very large files
        return hashlib.md5(audio_data[:102400]).hexdigest()
    
    async def _transcribe_whisper_local(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using local Whisper model."""
        if not WHISPER_AVAILABLE:
            raise ProviderError("Whisper", "Whisper is not installed")
        
        try:
            # Lazy load the model if not already loaded
            if not hasattr(self, '_whisper_model') or self._whisper_model is None:
                logger.info(f"Loading Whisper model ({settings.WHISPER_MODEL_SIZE})...")
                self._whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE)
            
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Run in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self._whisper_model.transcribe(
                    temp_path,
                    language="en",
                    temperature=0.0,
                    word_timestamps=False
                )
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Get transcription and confidence
            text = result["text"].strip()
            confidence = 0.0
            
            # Estimate confidence from segment scores
            if "segments" in result and result["segments"]:
                confidences = [seg.get("confidence", 0.0) for seg in result["segments"]]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            return text, confidence
        
        except Exception as e:
            raise ProviderError("Whisper", str(e))
    
    async def _transcribe_speechrecognition(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using SpeechRecognition library."""
        try:
            recognizer = sr.Recognizer()
            
            # Convert bytes to AudioData
            with io.BytesIO(audio_data) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
            
            # Try multiple recognition engines
            engines = [
                ("sphinx", lambda: recognizer.recognize_sphinx(audio_data)),
                ("google", lambda: recognizer.recognize_google(audio_data)),
                ("wit", lambda: recognizer.recognize_wit(audio_data, key=os.getenv("WIT_AI_KEY"))),
                ("azure", lambda: recognizer.recognize_azure(
                    audio_data, 
                    key=settings.AZURE_SPEECH_KEY,
                    location=settings.AZURE_SPEECH_REGION
                )),
                ("houndify", lambda: recognizer.recognize_houndify(
                    audio_data,
                    client_id=os.getenv("HOUNDIFY_CLIENT_ID"),
                    client_key=os.getenv("HOUNDIFY_CLIENT_KEY")
                )),
            ]
            
            # Try each engine until one works
            for engine_name, engine_func in engines:
                try:
                    text = engine_func()
                    if text:
                        return text, 0.8  # Assume decent confidence
                except:
                    continue
            
            # If all fail, use sphinx (offline)
            return recognizer.recognize_sphinx(audio_data), 0.6
        
        except sr.UnknownValueError:
            raise ProviderError("SpeechRecognition", "Speech not understood")
        except sr.RequestError as e:
            raise ProviderError("SpeechRecognition", f"Request error: {str(e)}")
        except Exception as e:
            raise ProviderError("SpeechRecognition", str(e))
    
    async def _transcribe_google(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using Google Cloud Speech-to-Text."""
        if not GOOGLE_CLOUD_SPEECH_AVAILABLE:
            raise ProviderError("Google", "Google Cloud Speech is not installed")
        
        try:
            # Initialize the client
            client = speech.SpeechClient()
            
            # Configure the request
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=settings.SAMPLE_RATE,
                language_code=settings.GOOGLE_LANGUAGE_CODE,
                enable_automatic_punctuation=True,
                model="latest_long",
                use_enhanced=True,
            )
            
            # Create the audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform the transcription
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.recognize(config=config, audio=audio)
            )
            
            # Process the response
            text = ""
            confidence = 0.0
            
            if response.results:
                # Combine all results
                for result in response.results:
                    text += result.alternatives[0].transcript + " "
                text = text.strip()
                
                # Calculate average confidence
                confidences = [result.alternatives[0].confidence for result in response.results if result.alternatives]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            return text, confidence
        
        except Exception as e:
            raise ProviderError("Google", str(e))
    
    async def _transcribe_azure(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using Azure Speech Services."""
        if not AZURE_SPEECH_AVAILABLE:
            raise ProviderError("Azure", "Azure Speech Services is not installed")
        
        try:
            # Initialize the speech config
            speech_config = azure_speech.SpeechConfig(
                subscription=settings.AZURE_SPEECH_KEY,
                region=settings.AZURE_SPEECH_REGION
            )
            
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Create the audio config
            audio_config = azure_speech.audio.AudioConfig(filename=temp_path)
            
            # Create the speech recognizer
            speech_recognizer = azure_speech.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Define the callback functions for events
            future = asyncio.Future()
            result_text = ""
            result_confidence = 0.0
            
            def recognized_cb(evt):
                nonlocal result_text, result_confidence
                result_text = evt.result.text
                result_confidence = 0.8  # Azure doesn't provide confidence scores directly
                future.set_result(True)
            
            def canceled_cb(evt):
                future.set_exception(ProviderError("Azure", f"Recognition canceled: {evt.result.reason}"))
            
            # Connect callbacks
            speech_recognizer.recognized.connect(recognized_cb)
            speech_recognizer.canceled.connect(canceled_cb)
            
            # Start recognition
            speech_recognizer.start_continuous_recognition()
            
            # Wait for the result
            try:
                await asyncio.wait_for(future, timeout=settings.TIMEOUT)
            finally:
                # Stop recognition and cleanup
                speech_recognizer.stop_continuous_recognition()
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return result_text, result_confidence
        
        except Exception as e:
            raise ProviderError("Azure", str(e))
    
    async def _transcribe_vosk(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using Vosk."""
        if not VOSK_AVAILABLE:
            raise ProviderError("Vosk", "Vosk is not installed")
        
        try:
            # Lazy load the model
            if not hasattr(self, '_vosk_model') or self._vosk_model is None:
                logger.info("Loading Vosk model...")
                self._vosk_model = Model(settings.VOSK_MODEL_PATH)
            
            # Convert bytes to wav data
            with io.BytesIO(audio_data) as audio_file:
                with wave.open(audio_file, "rb") as wf:
                    # Create recognizer
                    rec = KaldiRecognizer(self._vosk_model, wf.getframerate())
                    
                    # Process audio
                    result_text = ""
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            # Partial results
                            continue
                    
                    # Get final result
                    result = rec.FinalResult()
                    import json
                    result_json = json.loads(result)
                    result_text = result_json.get("text", "")
            
            return result_text, 0.7  # Vosk doesn't provide confidence scores
        
        except Exception as e:
            raise ProviderError("Vosk", str(e))
    
    async def _transcribe_deepspeech(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using DeepSpeech."""
        if not DEEPSPEECH_AVAILABLE:
            raise ProviderError("DeepSpeech", "DeepSpeech is not installed")
        
        try:
            # Lazy load the model
            if not hasattr(self, '_deepspeech_model') or self._deepspeech_model is None:
                logger.info("Loading DeepSpeech model...")
                self._deepspeech_model = deepspeech.Model(settings.DEEPSPEECH_MODEL_PATH)
            
            # Convert bytes to audio data
            with io.BytesIO(audio_data) as audio_file:
                with wave.open(audio_file, "rb") as wf:
                    # Get audio parameters
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    frames = wf.getnframes()
                    
                    # Read all audio data
                    buffer = wf.readframes(frames)
                    
                    # Convert to 16-bit mono if needed
                    if channels != 1:
                        # This is a simplistic conversion, ideally use a proper audio library
                        audio = np.frombuffer(buffer, dtype=np.int16).reshape(-1, channels)
                        audio = audio.mean(axis=1).astype(np.int16)
                        buffer = audio.tobytes()
            
            # Process with DeepSpeech
            loop = asyncio.get_event_loop()
            result_text = await loop.run_in_executor(
                None,
                lambda: self._deepspeech_model.stt(buffer)
            )
            
            return result_text, 0.7  # DeepSpeech doesn't provide confidence scores
        
        except Exception as e:
            raise ProviderError("DeepSpeech", str(e))
    
    async def _transcribe_whisper_api(self, audio_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            import openai
            
            # Configure API key
            openai.api_key = settings.WHISPER_API_KEY
            
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Open the file for the API request
            with open(temp_path, "rb") as audio_file:
                # Make the API request
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: openai.Audio.transcribe(
                        file=audio_file,
                        model="whisper-1",
                        language="en"
                    )
                )
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Extract text
            text = response.get("text", "").strip()
            
            return text, 0.9  # Whisper API doesn't provide confidence scores
        
        except Exception as e:
            raise ProviderError("Whisper API", str(e))
    
    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=settings.RETRY_DELAY, max=10),
        reraise=True
    )
    async def transcribe(self, audio_data: bytes, provider: Optional[STTProvider] = None) -> Tuple[str, float]:
        """
        Transcribe audio data using the specified or default provider.
        Falls back to other providers if the specified provider fails.
        
        Args:
            audio_data: Audio data as bytes
            provider: Optional specific provider to use
            
        Returns:
            Tuple of (transcription text, confidence score)
            
        Raises:
            STTError: If all providers fail
        """
        # Reset provider errors
        self.provider_errors = {}
        
        # Check the cache first if enabled
        if settings.ENABLE_CACHE and self.cache:
            cache_key = self._get_cache_key(audio_data)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Using cached transcription result")
                return cached_result
        
        # Preprocess the audio
        processed_audio, metadata = self._preprocess_audio(audio_data)
        
        # Determine which providers to try
        providers_to_try = []
        
        if provider and provider in self.available_providers:
            # If a specific provider is requested and available, try it first
            providers_to_try.append(provider)
        
        # Then add the default provider if it's not already in the list
        if settings.DEFAULT_STT_PROVIDER in self.available_providers and settings.DEFAULT_STT_PROVIDER not in providers_to_try:
            providers_to_try.append(settings.DEFAULT_STT_PROVIDER)
        
        # Then add fallback providers if they're not already in the list
        for fallback in settings.FALLBACK_STT_PROVIDERS:
            if fallback in self.available_providers and fallback not in providers_to_try:
                providers_to_try.append(fallback)
        
        # Finally, add any other available providers not already in the list
        for available in self.available_providers:
            if available not in providers_to_try:
                providers_to_try.append(available)
        
        # If no providers are available, raise an error
        if not providers_to_try:
            raise STTError("No STT providers are available. Check configuration.")
        
        # Try each provider in order
        for provider in providers_to_try:
            try:
                if provider == STTProvider.WHISPER_LOCAL:
                    text, confidence = await self._transcribe_whisper_local(processed_audio, metadata)
                elif provider == STTProvider.WHISPER_API:
                    text, confidence = await self._transcribe_whisper_api(processed_audio, metadata)
                elif provider == STTProvider.GOOGLE:
                    text, confidence = await self._transcribe_google(processed_audio, metadata)
                elif provider == STTProvider.AZURE:
                    text, confidence = await self._transcribe_azure(processed_audio, metadata)
                elif provider == STTProvider.VOSK:
                    text, confidence = await self._transcribe_vosk(processed_audio, metadata)
                elif provider == STTProvider.DEEPSPEECH:
                    text, confidence = await self._transcribe_deepspeech(processed_audio, metadata)
                elif provider == STTProvider.SPEECHRECOGNITION:
                    text, confidence = await self._transcribe_speechrecognition(processed_audio, metadata)
                else:
                    # Skip unsupported providers
                    logger.warning(f"Unsupported provider: {provider}")
                    continue
                
                # If we got a result, cache it and return
                if text:
                    result = (text, confidence)
                    
                    # Cache the result if enabled
                    if settings.ENABLE_CACHE and self.cache:
                        cache_key = self._get_cache_key(audio_data)
                        self.cache.set(cache_key, result, expire=settings.CACHE_TTL)
                    
                    return result
                else:
                    # If no text was returned, log and try the next provider
                    logger.warning(f"Provider {provider} returned no text")
                    self.provider_errors[provider.value] = "No transcription returned"
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
_multi_stt_client = None

def get_multi_stt_client() -> MultiSTTClient:
    """Get the multi-STT client singleton instance."""
    global _multi_stt_client
    if _multi_stt_client is None:
        _multi_stt_client = MultiSTTClient()
    return _multi_stt_client


async def transcribe(audio_data: bytes, provider: Optional[STTProvider] = None) -> Tuple[str, float]:
    """
    Transcribe audio data using the multi-STT client.
    
    Args:
        audio_data: Audio data as bytes
        provider: Optional specific provider to use
        
    Returns:
        Tuple of (transcription text, confidence score)
        
    Raises:
        STTError: If an error occurs in transcription
    """
    client = get_multi_stt_client()
    return await client.transcribe(audio_data, provider)
