# agents/voice_agent/multi_tts_client.py

"""
Multi-provider Text-to-Speech client that supports multiple TTS services with fallback mechanisms.
"""
import os
import io
import logging
import asyncio
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib # For cache key

# Caching
from diskcache import Cache

# Retry utilities
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
)

# Audio processing
from pydub import AudioSegment # For speed adjustment if provider doesn't support it

# TTS provider SDKs/libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.getLogger(__name__).warning("pyttsx3 library not found. pyttsx3 provider will be unavailable.")

try:
    from gtts import gTTS, gTTSError
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.getLogger(__name__).warning("gTTS library not found. gtts provider will be unavailable.")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.getLogger(__name__).warning("edge-tts library not found. edge provider will be unavailable.")

try:
    from elevenlabs import generate as elevenlabs_generate, set_api_key as elevenlabs_set_api_key, voices as elevenlabs_voices, Voice as ElevenLabsVoice, VoiceSettings as ElevenLabsVoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logging.getLogger(__name__).warning("elevenlabs library not found. elevenlabs provider will be unavailable.")

try:
    import boto3
    AWS_POLLY_AVAILABLE = True
except ImportError:
    AWS_POLLY_AVAILABLE = False
    logging.getLogger(__name__).warning("boto3 library not found. amazon-polly provider will be unavailable.")

try:
    import torch # For Silero
    # Silero models are typically loaded via torch.hub or local paths
    SILERO_AVAILABLE = True 
except ImportError:
    SILERO_AVAILABLE = False
    logging.getLogger(__name__).warning("torch library not found. silero provider will be unavailable.")

try:
    from TTS.api import TTS as CoquiTTS_API # Renamed to avoid conflict
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.getLogger(__name__).warning("TTS (Coqui) library not found. coqui provider will be unavailable.")


from .config import settings, TTSProvider

logger = logging.getLogger(__name__)


class TTSError(Exception): pass
class ProviderError(TTSError):
    def __init__(self, provider: str, message: str, original_exception: Optional[Exception] = None):
        self.provider = provider
        self.message = message
        self.original_exception = original_exception
        super().__init__(f"Provider '{provider}' error: {message}" + (f" (Original: {type(original_exception).__name__})" if original_exception else ""))
class AllProvidersFailedError(TTSError):
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors
        error_summary = "; ".join([f"'{p}': {e[:100]}..." if len(e) > 100 else f"'{p}': {e}" for p, e in provider_errors.items()])
        super().__init__(f"All TTS providers failed. Error summary: {error_summary}")


class MultiTTSClient:
    def __init__(self):
        self.provider_errors: Dict[str, str] = {}
        # Lazy-loaded models/clients
        self._pyttsx3_engine: Optional[pyttsx3.Engine] = None
        self._polly_client: Optional[Any] = None # boto3.client("polly")
        self._silero_model: Optional[Any] = None # torch model
        self._silero_speaker: str = settings.SILERO_SPEAKER
        self._silero_sample_rate: int = 48000 # Common for Silero, adjust if specific model differs
        self._coqui_tts_instance: Optional[CoquiTTS_API] = None

        if settings.ENABLE_CACHE:
            self.cache = Cache(settings.CACHE_DIR, tag_index=True, size_limit=int(1e9))
        else:
            self.cache = None
            
        self.available_providers = self._initialize_and_verify_providers()
        if not self.available_providers:
            logger.critical("CRITICAL: No TTS providers are available/configured. TTS functionality will be severely limited or non-functional.")

    def _initialize_and_verify_providers(self) -> List[TTSProvider]:
        truly_available = []
        sdk_availability_map = {
            TTSProvider.PYTTSX3: PYTTSX3_AVAILABLE,
            TTSProvider.GTTS: GTTS_AVAILABLE,
            TTSProvider.EDGE: EDGE_TTS_AVAILABLE,
            TTSProvider.ELEVENLABS: ELEVENLABS_AVAILABLE,
            TTSProvider.AMAZON_POLLY: AWS_POLLY_AVAILABLE,
            TTSProvider.SILERO: SILERO_AVAILABLE,
            TTSProvider.COQUI: COQUI_AVAILABLE,
        }
        for provider_enum in TTSProvider:
            is_configured = settings.is_tts_provider_available(provider_enum) # Checks config (keys, paths)
            sdk_present = sdk_availability_map.get(provider_enum, False)

            if is_configured and sdk_present:
                can_add = True
                # Further checks for API key based providers
                if provider_enum == TTSProvider.ELEVENLABS and not settings.ELEVENLABS_API_KEY:
                    logger.warning(f"{provider_enum.value}: Configured but ELEVENLABS_API_KEY missing.")
                    can_add = False
                elif provider_enum == TTSProvider.AMAZON_POLLY and not (settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY):
                    logger.warning(f"{provider_enum.value}: Configured but AWS credentials missing.")
                    can_add = False
                
                if can_add:
                    if provider_enum == TTSProvider.ELEVENLABS: # Set API key early
                        try: elevenlabs_set_api_key(settings.ELEVENLABS_API_KEY)
                        except Exception as e: logger.error(f"Failed to set ElevenLabs API key: {e}"); can_add=False
                    
                    if provider_enum == TTSProvider.AMAZON_POLLY:
                        try: # Initialize Polly client eagerly
                            self._polly_client = boto3.client(
                                'polly',
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                region_name=settings.AWS_REGION
                            )
                            logger.info("Amazon Polly client initialized.")
                        except Exception as e: logger.error(f"Failed to init Polly client: {e}. Disabled."); can_add=False
                
                if can_add:
                    truly_available.append(provider_enum)

            elif is_configured and not sdk_present:
                logger.warning(f"TTS Provider '{provider_enum.value}' configured but SDK/library missing. Unavailable.")
        
        logger.info(f"Final list of available TTS providers: {[p.value for p in truly_available]}")
        return truly_available

    def _get_cache_key(self, text: str, voice: Optional[str], speed: float, provider_val: str, lang: Optional[str]) -> str:
        m = hashlib.md5()
        m.update(text.encode())
        if voice: m.update(voice.encode())
        m.update(str(speed).encode())
        m.update(provider_val.encode())
        if lang: m.update(lang.encode())
        return m.hexdigest()

    async def _run_with_retry(self, provider_name_str: str, async_target_callable, *args):
        retryer = AsyncRetrying(
            stop=stop_after_attempt(settings.MAX_RETRIES),
            wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=1, max=10),
            reraise=True,
        )
        try:
            return await retryer.call(async_target_callable, *args)
        except asyncio.TimeoutError as e_timeout:
            msg = f"Request timed out after {settings.TIMEOUT}s (including retries)"
            logger.warning(f"{provider_name_str}: {msg}")
            raise ProviderError(provider_name_str, msg, e_timeout)
        except Exception as e:
            msg = f"Failed after {settings.MAX_RETRIES} retries: {str(e)}"
            logger.warning(f"{provider_name_str}: {msg} (Type: {type(e).__name__})")
            raise ProviderError(provider_name_str, msg, e)

    # --- Provider Specific Synthesis Methods ---
    async def _synthesize_pyttsx3(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not PYTTSX3_AVAILABLE: raise ProviderError(TTSProvider.PYTTSX3.value, "pyttsx3 SDK missing.")
        if self._pyttsx3_engine is None:
            logger.info("Initializing pyttsx3 engine...")
            try: self._pyttsx3_engine = pyttsx3.init()
            except Exception as e: raise ProviderError(TTSProvider.PYTTSX3.value, f"Engine init failed: {e}", e)

        engine = self._pyttsx3_engine
        if voice and voice != "default":
            available_voices = engine.getProperty('voices')
            chosen_voice = next((v for v in available_voices if voice.lower() in v.name.lower() or voice.lower() in v.id.lower()), None)
            if chosen_voice: engine.setProperty('voice', chosen_voice.id)
            else: logger.warning(f"pyttsx3: Voice '{voice}' not found. Using default.")
        
        current_rate = engine.getProperty('rate') # Default is often 200
        engine.setProperty('rate', int(current_rate * speed)) # pyttsx3 speed is words per minute

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            wav_path = tmp_f.name
        try:
            await asyncio.get_event_loop().run_in_executor(None, engine.save_to_file, text, wav_path)
            await asyncio.get_event_loop().run_in_executor(None, engine.runAndWait) # Necessary to complete saving
            
            with open(wav_path, "rb") as f_wav: audio_bytes = f_wav.read()
            # Convert WAV to MP3
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            mp3_bio = io.BytesIO()
            audio_segment.export(mp3_bio, format="mp3")
            return mp3_bio.getvalue()
        finally:
            if os.path.exists(wav_path): os.unlink(wav_path)

    async def _synthesize_gtts(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not GTTS_AVAILABLE: raise ProviderError(TTSProvider.GTTS.value, "gTTS SDK missing.")
        # gTTS voice param is language. 'voice' here can be 'en', 'fr', etc.
        # Or specific accent like 'en-uk', 'en-au'. gTTS handles this via `lang` and `tld`.
        effective_lang = lang or (voice if voice and len(voice) <= 5 else "en") # Simple lang detection from voice
        is_slow = speed < 0.85 # gTTS only has slow or normal

        try:
            tts_instance = gTTS(text=text, lang=effective_lang, slow=is_slow)
            mp3_bio = io.BytesIO()
            await asyncio.get_event_loop().run_in_executor(None, tts_instance.write_to_fp, mp3_bio)
            mp3_bio.seek(0)
            audio_bytes = mp3_bio.getvalue()

            # If speed needs more granular control than gTTS slow/normal, use pydub
            if not is_slow and speed != 1.0: # If not already slowed by gTTS and speed is not normal
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                # Pydub speedup changes pitch. For TTS, changing frame_rate to simulate speed is more common
                # but can sound unnatural. True speed change without pitch needs more complex DSP.
                # For now, let's use pydub's speedup if available, or skip granular speed for gTTS.
                # A simple frame rate change (less ideal):
                # audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={
                #    "frame_rate": int(audio_segment.frame_rate * speed)
                # })
                # Using speedup:
                if speed > 1.0: audio_segment = audio_segment.speedup(playback_speed=speed, chunk_size=150, crossfade=75)
                # Slowing down is harder with pydub without pitch issues.
                # For now, only supporting speed > 1.0 adjustment for gTTS via pydub.
                
                final_bio = io.BytesIO()
                audio_segment.export(final_bio, format="mp3")
                audio_bytes = final_bio.getvalue()
            return audio_bytes
        except gTTSError as e_gtts:
            raise ProviderError(TTSProvider.GTTS.value, f"gTTS API error: {e_gtts.msg}", e_gtts)
        except Exception as e:
            raise ProviderError(TTSProvider.GTTS.value, f"Synthesis failed: {e}", e)


    async def _synthesize_edge_tts(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not EDGE_TTS_AVAILABLE: raise ProviderError(TTSProvider.EDGE.value, "edge-tts SDK missing.")
        
        effective_voice = voice or "en-US-AriaNeural" # EdgeTTS default
        # Edge TTS rate is like "+20%" or "-10%". Convert speed (0.25-4.0) to this.
        # 1.0 speed = +0%.  2.0 speed = +100%. 0.5 speed = -50%.
        rate_modifier = int((speed - 1.0) * 100)
        rate_str = f"{rate_modifier:+}%" # Format with sign, e.g., "+20%", "-10%"

        mp3_bio = io.BytesIO()
        try:
            communicate = edge_tts.Communicate(text, effective_voice, rate=rate_str)
            # edge_tts.Communicate.stream() is an async generator
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bio.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass # logger.debug(f"EdgeTTS Word: {chunk['text']}")
            mp3_bio.seek(0)
            return mp3_bio.getvalue()
        except Exception as e: # Catch edge_tts specific errors if any, or general ones
             raise ProviderError(TTSProvider.EDGE.value, f"Synthesis failed: {e}", e)


    async def _synthesize_elevenlabs(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not (ELEVENLABS_AVAILABLE and settings.ELEVENLABS_API_KEY): 
            raise ProviderError(TTSProvider.ELEVENLABS.value, "ElevenLabs SDK/API Key missing.")
        
        effective_voice_id = voice or settings.ELEVENLABS_VOICE_ID or "Rachel" # Default if nothing else
        
        # ElevenLabs speed is 'stability' and 'similarity_boost' more than direct speed.
        # We can use stability for a proxy if speed is very low or high.
        # stability (0 to 1). Lower stability = more variable, higher = more monotonous.
        # similarity_boost (0 to 1).
        # For simplicity, direct speed mapping isn't clean. We'll use defaults.
        # If speed param is crucial, SSML <prosody rate="..."> might be an option if supported.
        # For now, not mapping 'speed' directly to ElevenLabs parameters.
        logger.debug(f"ElevenLabs speed param ({speed}) not directly mapped. Using default voice settings.")

        try:
            # elevenlabs_generate is synchronous
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: elevenlabs_generate(
                    text=text,
                    voice=ElevenLabsVoice(
                        voice_id=effective_voice_id,
                        # Optionally define settings here if voice object doesn't have them
                        # settings=ElevenLabsVoiceSettings(stability=0.75, similarity_boost=0.75) 
                    ),
                    model="eleven_multilingual_v2" # Or other suitable model
                )
            )
            if not isinstance(audio_bytes, bytes): # generate can return an iterator
                accumulated_bytes = bytearray()
                for chunk in audio_bytes: accumulated_bytes.extend(chunk)
                audio_bytes = bytes(accumulated_bytes)
            return audio_bytes
        except Exception as e:
            raise ProviderError(TTSProvider.ELEVENLABS.value, f"API call failed: {e}", e)


    async def _synthesize_amazon_polly(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not (AWS_POLLY_AVAILABLE and self._polly_client): 
            raise ProviderError(TTSProvider.AMAZON_POLLY.value, "Polly SDK/Client missing or not init.")

        effective_voice_id = voice or settings.POLLY_VOICE_ID
        # Polly uses SSML for speed. <prosody rate="x-slow|slow|medium|fast|x-fast|N%">
        # Mapping our float speed (0.25-4.0) to Polly's percentage (approx 25% to 400%)
        polly_rate_percent = int(speed * 100)
        
        # Ensure rate is within Polly's typical useful range (e.g. 20% to 200% for naturalness)
        polly_rate_percent_clamped = max(20, min(polly_rate_percent, 200)) 
        
        ssml_text = f'<speak><prosody rate="{polly_rate_percent_clamped}%">{text}</prosody></speak>'
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._polly_client.synthesize_speech(
                    Text=ssml_text,
                    TextType="ssml",
                    OutputFormat="mp3",
                    VoiceId=effective_voice_id,
                    LanguageCode=lang, # Optional, Polly infers from voice often
                    Engine="neural" # Or "standard"
                )
            )
            if "AudioStream" in response:
                # The stream needs to be read. Boto3 HttpBody is a stream.
                audio_stream = response['AudioStream']
                audio_bytes = audio_stream.read()
                audio_stream.close()
                return audio_bytes
            raise ProviderError(TTSProvider.AMAZON_POLLY.value, "No AudioStream in Polly response.")
        except Exception as e:
            raise ProviderError(TTSProvider.AMAZON_POLLY.value, f"API call failed: {e}", e)

    async def _synthesize_silero(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not SILERO_AVAILABLE: raise ProviderError(TTSProvider.SILERO.value, "Silero (torch) SDK missing.")
        if self._silero_model is None:
            logger.info("Loading Silero TTS model...")
            # Silero V3/V4 models recommend specific sample rates
            # For V3, common sample rates are 8kHz, 24kHz, 48kHz. Let's use 48kHz for quality.
            self._silero_sample_rate = 48000 
            model_name = "v3_en" # Example V3 English model
            # Silero models typically are just `model.pt`
            silero_model_file = Path(settings.SILERO_MODEL_PATH) / f"{model_name}.pt"
            try:
                if silero_model_file.exists():
                    self._silero_model = torch.package.PackageImporter(str(silero_model_file)).load_pickle("tts_models", "model")
                else: # Fallback to torch.hub
                    logger.info(f"Local Silero model {silero_model_file} not found, trying torch.hub an 'en' model")
                    # This part needs internet and might download a large model on first run.
                    self._silero_model, _ = await asyncio.get_event_loop().run_in_executor(
                         None, lambda: torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=lang or settings.SILERO_LANGUAGE, speaker=settings.SILERO_SPEAKER_V3_IDS[0])) # Default to first available v3 speaker
                # Assuming model is a new Silero format that takes **kwargs
                # self._silero_model.to(torch.device('cpu')) # Ensure CPU if GPU not intended/available
            except Exception as e: raise ProviderError(TTSProvider.SILERO.value, f"Silero model load failed: {e}", e)

        effective_speaker = voice or settings.SILERO_SPEAKER # e.g. "en_0", "random"
        
        # Silero `apply_tts` has `speed` parameter (float, 1.0 is normal)
        # For older models, it might be sample_rate adjustment.
        # Newer models `save_wav` or `apply_tts` may have speed/rate directly.
        # Assuming `apply_tts` exists and handles it, or we post-process.
        
        audio_tensor = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._silero_model.apply_tts(
                text=text,
                speaker=effective_speaker,
                sample_rate=self._silero_sample_rate,
                # put_accent=True, put_yo=True, # Example extra params for some models
                speed=speed # Pass speed directly if model supports it
            )
        )
        # audio_tensor is a 1D torch tensor. Convert to WAV bytes then MP3.
        wav_bio = io.BytesIO()
        # torchaudio.save needs torchaudio. torchaudio.save(wav_bio, audio_tensor.unsqueeze(0).cpu(), self._silero_sample_rate, format="wav")
        # Simpler: convert tensor to numpy, then to AudioSegment
        audio_np = audio_tensor.numpy()
        # Scale to int16 if it's float -1 to 1
        if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
            audio_np = (audio_np * 32767).astype(np.int16)
        
        audio_segment = AudioSegment(data=audio_np.tobytes(), sample_width=2, frame_rate=self._silero_sample_rate, channels=1)
        mp3_bio = io.BytesIO()
        audio_segment.export(mp3_bio, format="mp3")
        return mp3_bio.getvalue()


    async def _synthesize_coqui(self, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        if not COQUI_AVAILABLE: raise ProviderError(TTSProvider.COQUI.value, "Coqui TTS SDK missing.")
        if self._coqui_tts_instance is None:
            logger.info(f"Loading Coqui TTS model: {settings.COQUI_MODEL_PATH or settings.COQUI_DEFAULT_MODEL_NAME}")
            model_source = settings.COQUI_MODEL_PATH if settings.COQUI_MODEL_PATH and Path(settings.COQUI_MODEL_PATH).exists() else settings.COQUI_DEFAULT_MODEL_NAME
            try:
                # Coqui TTS API can take model_path or model_name
                if Path(model_source).is_dir() or Path(model_source).is_file() : # is local path
                    self._coqui_tts_instance = CoquiTTS_API(model_path=model_source, progress_bar=False)
                else: # is model name from their list
                    self._coqui_tts_instance = CoquiTTS_API(model_name=model_source, progress_bar=False)
            except Exception as e: raise ProviderError(TTSProvider.COQUI.value, f"Coqui model load failed ({model_source}): {e}", e)
        
        # Coqui tts_to_file or tts method. `tts` returns a list of int samples (PCM).
        # Speaker can be by ID, path to speaker embedding, or None for default.
        # Speed is not a direct param for coqui `tts` method. Post-processing needed.
        speaker_arg = voice if voice and voice != "default" else None # Use default if not specified
        language_arg = lang if lang else None # Coqui models are often language-specific

        wav_samples_list = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._coqui_tts_instance.tts(text=text, speaker=speaker_arg, language=language_arg)
        )
        # wav_samples_list is List[int] usually. Convert to numpy int16 array.
        wav_np = np.array(wav_samples_list, dtype=np.int16)
        
        # Coqui models have their own sample rate, get it from the model
        coqui_sr = self._coqui_tts_instance.synthesizer.output_sample_rate if self._coqui_tts_instance.synthesizer else 22050 # Default fallback

        audio_segment = AudioSegment(data=wav_np.tobytes(), sample_width=2, frame_rate=coqui_sr, channels=1)

        # Apply speed adjustment if needed (post-processing)
        if speed != 1.0:
            logger.debug(f"Coqui: Applying speed {speed} post-synthesis.")
            # This basic speed change affects pitch.
            # audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": int(coqui_sr * speed)})
            if speed > 1.0: audio_segment = audio_segment.speedup(playback_speed=speed, chunk_size=150, crossfade=75)
            # Slowdown is harder.
        
        mp3_bio = io.BytesIO()
        audio_segment.export(mp3_bio, format="mp3")
        return mp3_bio.getvalue()


    async def _dispatch_synthesize(self, provider: TTSProvider, text: str, voice: Optional[str], speed: float, lang: Optional[str]) -> bytes:
        dispatch_map = {
            TTSProvider.PYTTSX3: self._synthesize_pyttsx3,
            TTSProvider.GTTS: self._synthesize_gtts,
            TTSProvider.EDGE: self._synthesize_edge_tts,
            TTSProvider.ELEVENLABS: self._synthesize_elevenlabs,
            TTSProvider.AMAZON_POLLY: self._synthesize_amazon_polly,
            TTSProvider.SILERO: self._synthesize_silero,
            TTSProvider.COQUI: self._synthesize_coqui,
        }
        if provider in dispatch_map:
            return await dispatch_map[provider](text, voice, speed, lang)
        raise ProviderError(provider.value, "Provider synthesis logic not implemented.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0, 
                       provider_preference: Optional[TTSProvider] = None, language: Optional[str] = None) -> bytes:
        self.provider_errors.clear()
        
        effective_provider_for_cache = provider_preference if provider_preference else settings.DEFAULT_TTS_PROVIDER
        cache_key = self._get_cache_key(text, voice, speed, effective_provider_for_cache.value, language)
        
        if self.cache:
            cached_audio = self.cache.get(cache_key)
            if cached_audio: logger.debug(f"Cache hit TTS key {cache_key[:10]}..."); return cached_audio

        providers_to_try = self._get_provider_attempt_order(provider_preference)
        if not providers_to_try: raise TTSError("No TTS providers available.")

        logger.info(f"TTS: Order={[p.value for p in providers_to_try]}. Text: '{text[:30]}...'. Voice: {voice}, Speed: {speed}")

        for current_provider in providers_to_try:
            try:
                logger.info(f"TTS: Attempting provider: {current_provider.value}")
                # Retry logic applied by _run_with_retry wrapper on the _dispatch_synthesize call
                # This means self._dispatch_synthesize is the 'async_target_callable'
                audio_bytes = await self._run_with_retry(
                    current_provider.value, 
                    self._dispatch_synthesize, 
                    current_provider, text, voice, speed, language
                )

                if audio_bytes:
                    if self.cache: self.cache.set(cache_key, audio_bytes, expire=settings.CACHE_TTL, tag=current_provider.value)
                    logger.info(f"TTS: Synthesized with {current_provider.value}. Output bytes: {len(audio_bytes)}")
                    return audio_bytes
                else: # Should not happen if _dispatch_synthesize raises ProviderError on empty audio
                    msg = "Provider returned no audio data."
                    logger.warning(f"{current_provider.value}: {msg}")
                    self.provider_errors[current_provider.value] = msg
            
            except ProviderError as e_p: self.provider_errors[e_p.provider] = e_p.message; logger.warning(str(e_p))
            except Exception as e_unexp: logger.error(f"TTS: Unexpected error with {current_provider.value}: {e_unexp}", exc_info=True); self.provider_errors[current_provider.value] = f"Unexpected: {str(e_unexp)}"
        
        raise AllProvidersFailedError(self.provider_errors)

    def _get_provider_attempt_order(self, preference: Optional[TTSProvider]) -> List[TTSProvider]:
        order: List[TTSProvider] = []
        def add_to_order(p: TTSProvider):
            if p in self.available_providers and p not in order: order.append(p)
        
        if preference: add_to_order(preference)
        add_to_order(settings.DEFAULT_TTS_PROVIDER)
        for fb_p in settings.FALLBACK_TTS_PROVIDERS: add_to_order(fb_p)
        for avail_p in self.available_providers: add_to_order(avail_p)
        return order


_tts_client_instance: Optional[MultiTTSClient] = None
def get_multi_tts_client() -> MultiTTSClient:
    global _tts_client_instance
    if _tts_client_instance is None: _tts_client_instance = MultiTTSClient()
    return _tts_client_instance

async def synthesize(text: str, voice: Optional[str] = None, speed: float = 1.0, 
                     provider_preference: Optional[TTSProvider] = None, language: Optional[str] = None) -> bytes:
    client = get_multi_tts_client()
    return await client.synthesize(text, voice, speed, provider_preference, language)