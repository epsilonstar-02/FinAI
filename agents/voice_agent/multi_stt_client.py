# agents/voice_agent/multi_stt_client.py (Corrected Retry Logic within Transcribe)

"""
Multi-provider Speech-to-Text client that supports multiple STT services with fallback mechanisms.
Integrates VAD and denoising capabilities.
"""
import os
import io
import logging
import asyncio
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import wave 
import numpy as np
import json # For Vosk result parsing

# Caching
from diskcache import Cache

# Retry utilities
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    # RetryError # Not explicitly caught, but good to be aware of
)

# Audio processing
from pydub import AudioSegment
import webrtcvad

try:
    from rnnoise_wrapper import RNNoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False
    logging.getLogger(__name__).warning("RNNoise/rnnoise_wrapper not available. Denoising with RNNoise will be disabled.")

# STT provider SDKs/libraries (same conditional imports as before)
import speech_recognition as sr

try:
    import whisper as openai_whisper # Local Whisper
    WHISPER_LOCAL_AVAILABLE = True
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False
    logging.getLogger(__name__).warning("OpenAI Whisper (local) library not found. whisper-local provider will be unavailable.")

try:
    import openai # For Whisper API
    OPENAI_API_CLIENT_INSTALLED = True # Renamed to avoid conflict if openai is also whisper's name
except ImportError:
    OPENAI_API_CLIENT_INSTALLED = False
    logging.getLogger(__name__).warning("OpenAI API library (for Whisper API) not found. whisper-api provider will be unavailable.")


try:
    from google.cloud import speech as google_cloud_speech_sdk
    GOOGLE_CLOUD_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_SPEECH_AVAILABLE = False
    logging.getLogger(__name__).warning("Google Cloud Speech SDK not found. google provider will be unavailable.")

try:
    import azure.cognitiveservices.speech as azure_speech_sdk
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False
    logging.getLogger(__name__).warning("Azure Cognitive Services Speech SDK not found. azure provider will be unavailable.")

try:
    from vosk import Model as VoskModel, KaldiRecognizer as VoskKaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.getLogger(__name__).warning("Vosk library not found. vosk provider will be unavailable.")

try:
    import deepspeech as deepspeech_sdk
    DEEPSPEECH_AVAILABLE = True
except ImportError:
    DEEPSPEECH_AVAILABLE = False
    logging.getLogger(__name__).warning("DeepSpeech library not found. deepspeech provider will be unavailable.")


from .config import settings, STTProvider

logger = logging.getLogger(__name__)

class STTError(Exception): pass
class ProviderError(STTError):
    def __init__(self, provider: str, message: str, original_exception: Optional[Exception] = None):
        self.provider = provider
        self.message = message
        self.original_exception = original_exception
        super().__init__(f"Provider '{provider}' error: {message}" + (f" (Original: {type(original_exception).__name__})" if original_exception else ""))
class AllProvidersFailedError(STTError):
    def __init__(self, provider_errors: Dict[str, str]):
        self.provider_errors = provider_errors
        error_summary = "; ".join([f"'{p}': {e[:100]}..." if len(e) > 100 else f"'{p}': {e}" for p, e in provider_errors.items()])
        super().__init__(f"All STT providers failed. Error summary: {error_summary}")


class MultiSTTClient:
    def __init__(self):
        self.provider_errors: Dict[str, str] = {}
        self._whisper_local_model: Optional[openai_whisper.Whisper] = None
        self._vosk_model: Optional[VoskModel] = None
        self._deepspeech_model: Optional[deepspeech_sdk.Model] = None
        self._rnnoise_denoiser: Optional[RNNoise] = None
        self._google_speech_client: Optional[google_cloud_speech_sdk.SpeechClient] = None
        self._openai_api_client: Optional[openai.AsyncOpenAI] = None


        if settings.ENABLE_CACHE:
            self.cache = Cache(settings.CACHE_DIR, tag_index=True, size_limit=int(1e9)) # 1GB
        else:
            self.cache = None
        
        self.vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS) if settings.VAD_ENABLED else None

        if settings.RNNOISE_ENABLED and RNNOISE_AVAILABLE:
            try: self._rnnoise_denoiser = RNNoise()
            except Exception as e: logger.warning(f"Failed to init RNNoise: {e}. Disabled.")
        
        self.available_providers = self._initialize_and_verify_providers()
        if not self.available_providers:
            logger.critical("No STT providers available/configured.")

    def _initialize_and_verify_providers(self) -> List[STTProvider]:
        truly_available = []
        # Map enums to SDK availability flags for cleaner checks
        sdk_availability_map = {
            STTProvider.WHISPER_LOCAL: WHISPER_LOCAL_AVAILABLE,
            STTProvider.WHISPER_API: OPENAI_API_CLIENT_INSTALLED, # Uses the 'openai' lib
            STTProvider.GOOGLE: GOOGLE_CLOUD_SPEECH_AVAILABLE,
            STTProvider.AZURE: AZURE_SPEECH_AVAILABLE,
            STTProvider.VOSK: VOSK_AVAILABLE,
            STTProvider.DEEPSPEECH: DEEPSPEECH_AVAILABLE,
            STTProvider.SPEECHRECOGNITION: True, # sr library itself, specific engines vary
        }

        for provider_enum in STTProvider:
            is_configured = settings.is_stt_provider_available(provider_enum)
            sdk_present = sdk_availability_map.get(provider_enum, False)

            if is_configured and sdk_present:
                # Further checks for API key based providers or client initialization
                can_add = True
                if provider_enum == STTProvider.WHISPER_API and not settings.WHISPER_API_KEY:
                    logger.warning(f"{provider_enum.value}: Configured but WHISPER_API_KEY is missing.")
                    can_add = False
                elif provider_enum == STTProvider.GOOGLE:
                    if settings.GOOGLE_APPLICATION_CREDENTIALS:
                        try:
                            self._google_speech_client = google_cloud_speech_sdk.SpeechClient()
                            logger.info("Google Cloud Speech client initialized.")
                        except Exception as e:
                            logger.error(f"Failed to initialize Google Cloud Speech client (creds: {settings.GOOGLE_APPLICATION_CREDENTIALS}): {e}. It will be unavailable.")
                            can_add = False
                    else:
                        logger.warning(f"{provider_enum.value}: Configured but GOOGLE_APPLICATION_CREDENTIALS not set.")
                        can_add = False
                elif provider_enum == STTProvider.AZURE and not (settings.AZURE_SPEECH_KEY and settings.AZURE_SPEECH_REGION):
                    logger.warning(f"{provider_enum.value}: Configured but AZURE_SPEECH_KEY or AZURE_SPEECH_REGION missing.")
                    can_add = False
                
                if can_add:
                    truly_available.append(provider_enum)
                    
            elif is_configured and not sdk_present:
                logger.warning(f"STT Provider '{provider_enum.value}' configured but its SDK/library is missing. Unavailable.")
        
        if settings.WHISPER_API_KEY and OPENAI_API_CLIENT_INSTALLED:
            self._openai_api_client = openai.AsyncOpenAI(api_key=settings.WHISPER_API_KEY)


        logger.info(f"Final list of available STT providers: {[p.value for p in truly_available]}")
        return truly_available

    def _get_cache_key(self, audio_hash_part: bytes, provider_val: str, lang: Optional[str]) -> str:
        import hashlib
        m = hashlib.md5()
        m.update(audio_hash_part)
        m.update(provider_val.encode())
        if lang: m.update(lang.encode())
        return m.hexdigest()

    def _denoise_audio_if_enabled(self, audio_seg: AudioSegment) -> AudioSegment:
        # (Implementation from previous corrected response is fine)
        if not (settings.NOISE_REDUCTION_ENABLED and self._rnnoise_denoiser): return audio_seg
        try:
            req_sr, req_ch, req_sw = 48000, 1, 2 
            current_seg = audio_seg
            if current_seg.frame_rate != req_sr: current_seg = current_seg.set_frame_rate(req_sr)
            if current_seg.channels != req_ch: current_seg = current_seg.set_channels(req_ch)
            if current_seg.sample_width != req_sw: current_seg = current_seg.set_sample_width(req_sw)
            
            frame_dur_ms = 10 
            samples_per_frame = int(req_sr * (frame_dur_ms / 1000.0)) 
            bytes_per_frame = samples_per_frame * req_sw

            raw = current_seg.raw_data
            denoised_raw = bytearray()
            for i in range(0, len(raw), bytes_per_frame):
                chunk = raw[i : i + bytes_per_frame]
                if len(chunk) < bytes_per_frame: chunk += b'\0' * (bytes_per_frame - len(chunk))
                denoised_raw.extend(self._rnnoise_denoiser.process_frame(chunk))
            return AudioSegment(data=bytes(denoised_raw), sample_width=req_sw, frame_rate=req_sr, channels=req_ch)
        except Exception as e: logger.warning(f"RNNoise denoising failed: {e}. Original audio used.")
        return audio_seg


    def _vad_split(self, audio_seg: AudioSegment) -> List[AudioSegment]:
        # (Implementation from previous corrected response is fine, but simplify segment creation)
        if not self.vad or audio_seg.duration_seconds < 0.1: return [audio_seg] # Min duration for VAD
        
        vad_sr, vad_ch, vad_sw = 16000, 1, 2
        proc_seg = audio_seg
        if proc_seg.frame_rate not in [8000, 16000, 32000, 48000]: proc_seg = proc_seg.set_frame_rate(vad_sr)
        if proc_seg.channels != vad_ch: proc_seg = proc_seg.set_channels(vad_ch)
        if proc_seg.sample_width != vad_sw: proc_seg = proc_seg.set_sample_width(vad_sw)

        sr_actual_vad = proc_seg.frame_rate
        frame_dur_ms = 30 
        samples_per_frame = int(sr_actual_vad * (frame_dur_ms / 1000.0))
        bytes_per_frame = samples_per_frame * vad_sw

        raw_data = proc_seg.raw_data
        frames = [raw_data[i : i + bytes_per_frame] for i in range(0, len(raw_data), bytes_per_frame) if len(raw_data[i : i + bytes_per_frame]) == bytes_per_frame]
        
        is_speech_flags = [self.vad.is_speech(f, sr_actual_vad) for f in frames]
        
        speech_audio_segments = []
        current_speech_chunk_raw = bytearray()
        
        for i, is_speech in enumerate(is_speech_flags):
            if is_speech:
                current_speech_chunk_raw.extend(frames[i])
            elif len(current_speech_chunk_raw) > 0: # End of a speech segment
                speech_audio_segments.append(AudioSegment(data=bytes(current_speech_chunk_raw), sample_width=vad_sw, frame_rate=sr_actual_vad, channels=vad_ch))
                current_speech_chunk_raw = bytearray()
        
        if len(current_speech_chunk_raw) > 0: # Trailing speech
            speech_audio_segments.append(AudioSegment(data=bytes(current_speech_chunk_raw), sample_width=vad_sw, frame_rate=sr_actual_vad, channels=vad_ch))
            
        # If VAD produced segments, return them. Otherwise, return the original (processed for VAD) segment.
        return speech_audio_segments if speech_audio_segments else [proc_seg]


    async def _preprocess_audio(self, audio_data: bytes) -> Tuple[AudioSegment, List[AudioSegment]]:
        try: audio = AudioSegment.from_file(io.BytesIO(audio_data))
        except Exception as e: raise STTError(f"Invalid audio format: {e}")

        denoised = self._denoise_audio_if_enabled(audio)
        
        # Standardize for STT (e.g., 16kHz mono, 16-bit)
        stt_target_sr = settings.SAMPLE_RATE
        stt_target_ch = settings.CHANNELS
        stt_target_sw = 2 # 16-bit

        if denoised.frame_rate != stt_target_sr: denoised = denoised.set_frame_rate(stt_target_sr)
        if denoised.channels != stt_target_ch: denoised = denoised.set_channels(stt_target_ch)
        if denoised.sample_width != stt_target_sw: denoised = denoised.set_sample_width(stt_target_sw)

        vad_segments = self._vad_split(denoised)
        return denoised, vad_segments

    async def _transcribe_whisper_local_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        # (Same as corrected version)
        if not WHISPER_LOCAL_AVAILABLE: raise ProviderError(STTProvider.WHISPER_LOCAL.value, "Whisper (local) library not installed.")
        if self._whisper_local_model is None:
            logger.info(f"Loading Whisper (local) model: {settings.WHISPER_MODEL_SIZE} from {settings.WHISPER_MODEL_PATH}")
            Path(settings.WHISPER_MODEL_PATH).mkdir(parents=True, exist_ok=True)
            try: self._whisper_local_model = openai_whisper.load_model(settings.WHISPER_MODEL_SIZE, download_root=settings.WHISPER_MODEL_PATH)
            except Exception as e: raise ProviderError(STTProvider.WHISPER_LOCAL.value, f"Model load failed: {e}", e)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            audio_seg.export(tmp_f.name, format="wav")
            fp = tmp_f.name
        try:
            opts = {"language": lang, "fp16": False} if lang else {"fp16": False} # fp16=False for wider CPU compatibility
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: self._whisper_local_model.transcribe(fp, **opts))
        finally: os.unlink(fp)
        
        text = result.get("text", "").strip()
        logprobs = [s.get('avg_logprob', -np.inf) for s in result.get("segments", []) if 'avg_logprob' in s and s.get('no_speech_prob', 1.0) < 0.8]
        conf = np.exp(np.mean(logprobs)) if logprobs and text else 0.0
        return text, float(min(max(conf, 0.0), 1.0))


    async def _transcribe_whisper_api_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        if not (OPENAI_API_CLIENT_INSTALLED and self._openai_api_client): 
            raise ProviderError(STTProvider.WHISPER_API.value, "Whisper API SDK/Client missing or not init.")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            audio_seg.export(tmp_f.name, format="wav")
            fp_path = tmp_f.name
        try:
            with open(fp_path, "rb") as audio_file_obj:
                response = await self._openai_api_client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file_obj, language=lang
                )
        finally: os.unlink(fp_path)
        
        text = response.text.strip() if response and response.text else ""
        return text, 0.95 if text else 0.0

    async def _transcribe_google_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        if not (GOOGLE_CLOUD_SPEECH_AVAILABLE and self._google_speech_client): 
            raise ProviderError(STTProvider.GOOGLE.value, "Google STT SDK/Client missing or not init.")
        
        if audio_seg.sample_width != 2: audio_seg = audio_seg.set_sample_width(2)
        
        g_lang = lang or settings.GOOGLE_LANGUAGE_CODE
        config = google_cloud_speech_sdk.RecognitionConfig(
            encoding=google_cloud_speech_sdk.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=audio_seg.frame_rate,
            language_code=g_lang,
            enable_automatic_punctuation=True
        )
        g_audio = google_cloud_speech_sdk.RecognitionAudio(content=audio_seg.raw_data)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: self._google_speech_client.recognize(config=config, audio=g_audio, timeout=settings.TIMEOUT-5)) # slightly less timeout for API call
        except Exception as e:
            raise ProviderError(STTProvider.GOOGLE.value, f"API call failed: {e}", e)

        text, conf = "", 0.0
        if response.results and response.results[0].alternatives:
            best_alt = response.results[0].alternatives[0]
            text = best_alt.transcript.strip()
            conf = float(best_alt.confidence)
        return text, conf

    async def _transcribe_azure_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        if not (AZURE_SPEECH_AVAILABLE and settings.AZURE_SPEECH_KEY and settings.AZURE_SPEECH_REGION):
            raise ProviderError(STTProvider.AZURE.value, "Azure STT SDK/Config missing.")

        speech_config = azure_speech_sdk.SpeechConfig(subscription=settings.AZURE_SPEECH_KEY, region=settings.AZURE_SPEECH_REGION)
        if lang: speech_config.speech_recognition_language = lang

        # Azure SDK's PushAudioInputStream is good for in-memory data
        stream = azure_speech_sdk.audio.PushAudioInputStream(
            stream_format=azure_speech_sdk.audio.AudioStreamFormat(
                samples_per_second=audio_seg.frame_rate,
                bits_per_sample=audio_seg.sample_width * 8,
                channels=audio_seg.channels
            )
        )
        stream.write(audio_seg.raw_data)
        stream.close() # Signal end of stream

        audio_input = azure_speech_sdk.AudioConfig(stream=stream)
        recognizer = azure_speech_sdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, recognizer.recognize_once)
        except Exception as e:
            raise ProviderError(STTProvider.AZURE.value, f"recognize_once failed: {e}", e)

        text, conf = "", 0.0
        if result.reason == azure_speech_sdk.ResultReason.RecognizedSpeech:
            text = result.text.strip()
            conf = 0.9 # Default, as detailed confidence is harder with recognize_once
        elif result.reason == azure_speech_sdk.ResultReason.Canceled:
            cancel_details = result.cancellation_details
            raise ProviderError(STTProvider.AZURE.value, f"Recognition Canceled: {cancel_details.reason} - {cancel_details.error_details}")
        return text, conf

    async def _transcribe_vosk_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        if not VOSK_AVAILABLE: raise ProviderError(STTProvider.VOSK.value, "Vosk SDK missing.")
        if self._vosk_model is None:
            logger.info(f"Loading Vosk model from {settings.VOSK_MODEL_PATH}")
            model_path_obj = Path(settings.VOSK_MODEL_PATH)
            if not model_path_obj.exists() or not model_path_obj.is_dir(): # Vosk model path is a directory
                raise ProviderError(STTProvider.VOSK.value, f"Vosk model path not found or not a directory: {settings.VOSK_MODEL_PATH}")
            try: self._vosk_model = VoskModel(str(model_path_obj)) # Pass string path
            except Exception as e: raise ProviderError(STTProvider.VOSK.value, f"Vosk model load failed: {e}", e)

        rec = VoskKaldiRecognizer(self._vosk_model, audio_seg.frame_rate)
        rec.SetWords(True) 
        
        # Feed data in chunks
        chunk_size = 4000 # bytes
        for i in range(0, len(audio_seg.raw_data), chunk_size):
            rec.AcceptWaveform(audio_seg.raw_data[i : i + chunk_size])
        
        result_json = json.loads(rec.FinalResult())
        
        text = result_json.get("text", "").strip()
        word_confs = [item['conf'] for item in result_json.get('result', []) if 'conf' in item]
        conf = np.mean(word_confs) if word_confs and text else 0.0
        return text, float(conf)

    async def _transcribe_deepspeech_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        if not DEEPSPEECH_AVAILABLE: raise ProviderError(STTProvider.DEEPSPEECH.value, "DeepSpeech SDK missing.")
        if self._deepspeech_model is None:
            logger.info(f"Loading DeepSpeech model from {settings.DEEPSPEECH_MODEL_PATH}")
            model_dir = Path(settings.DEEPSPEECH_MODEL_PATH)
            # Common pattern for DeepSpeech model files
            model_file = next(model_dir.glob("*.pbmm"), None)
            scorer_file = next(model_dir.glob("*.scorer"), None)

            if not model_file or not model_file.is_file():
                raise ProviderError(STTProvider.DEEPSPEECH.value, f"DeepSpeech model file (.pbmm) not found in {model_dir}")
            try:
                self._deepspeech_model = deepspeech_sdk.Model(str(model_file))
                if scorer_file and scorer_file.is_file(): 
                    logger.info(f"Enabling DeepSpeech scorer: {scorer_file}")
                    self._deepspeech_model.enableExternalScorer(str(scorer_file))
                # Beam width, alpha, beta can be set here from settings if needed
                # self._deepspeech_model.setBeamWidth(500) 
            except Exception as e: raise ProviderError(STTProvider.DEEPSPEECH.value, f"DS model load failed: {e}", e)

        ds_audio_seg = audio_seg
        if ds_audio_seg.sample_width != 2: ds_audio_seg = ds_audio_seg.set_sample_width(2)
        if ds_audio_seg.channels != 1: ds_audio_seg = ds_audio_seg.set_channels(1)
        if ds_audio_seg.frame_rate != 16000: ds_audio_seg = ds_audio_seg.set_frame_rate(16000) # DS typically 16kHz
        
        audio_buffer = np.frombuffer(ds_audio_seg.raw_data, dtype=np.int16)
        
        # sttWithMetadata is synchronous
        result_meta = await asyncio.get_event_loop().run_in_executor(None, lambda: self._deepspeech_model.sttWithMetadata(audio_buffer, 1)) # num_results=1
        
        text = "".join(token.text for token in result_meta.transcripts[0].tokens).strip() if result_meta.transcripts else ""
        conf = result_meta.transcripts[0].confidence if result_meta.transcripts and text else 0.0
        return text, float(conf)

    async def _transcribe_speechrecognition_segment(self, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        # (Same as corrected version)
        recognizer = sr.Recognizer()
        with io.BytesIO() as bio:
            audio_seg.export(bio, format="wav")
            bio.seek(0)
            with sr.AudioFile(bio) as source:
                audio_data_sr = recognizer.record(source)
        
        lang_code = lang or "en-US"
        try:
            text = await asyncio.get_event_loop().run_in_executor(None, lambda: recognizer.recognize_google(audio_data_sr, language=lang_code))
            return text.strip(), 0.85 
        except sr.UnknownValueError: logger.debug("SR (Google) could not understand audio.")
        except sr.RequestError as e_google_req: logger.warning(f"SR (Google) request error: {e_google_req}.")
        
        try: # Fallback to Sphinx
            text = await asyncio.get_event_loop().run_in_executor(None, lambda: recognizer.recognize_sphinx(audio_data_sr, language=lang_code))
            return text.strip(), 0.60
        except sr.UnknownValueError: logger.debug("SR (Sphinx) could not understand audio.")
        except Exception as e_sphinx: raise ProviderError(STTProvider.SPEECHRECOGNITION.value, f"Sphinx (via SR) failed: {e_sphinx}", e_sphinx)
        return "", 0.0 # If both failed

    async def _dispatch_transcribe(self, provider: STTProvider, audio_seg: AudioSegment, lang: Optional[str]) -> Tuple[str, float]:
        # (Same as corrected version)
        dispatch_map = {
            STTProvider.WHISPER_LOCAL: self._transcribe_whisper_local_segment,
            STTProvider.WHISPER_API: self._transcribe_whisper_api_segment,
            STTProvider.GOOGLE: self._transcribe_google_segment,
            STTProvider.AZURE: self._transcribe_azure_segment,
            STTProvider.VOSK: self._transcribe_vosk_segment,
            STTProvider.DEEPSPEECH: self._transcribe_deepspeech_segment,
            STTProvider.SPEECHRECOGNITION: self._transcribe_speechrecognition_segment,
        }
        if provider in dispatch_map:
            return await dispatch_map[provider](audio_seg, lang)
        raise ProviderError(provider.value, "Provider transcription logic not implemented in _dispatch_transcribe.")

    async def transcribe(self, audio_data_bytes: bytes, 
                       provider_preference: Optional[STTProvider] = None, 
                       language: Optional[str] = None) -> Tuple[str, float]:
        self.provider_errors.clear()
        audio_hash_part = audio_data_bytes[:min(len(audio_data_bytes), 1024*100)] 
        effective_provider_for_cache = provider_preference if provider_preference else settings.DEFAULT_STT_PROVIDER
        # Ensure effective_provider_for_cache is a valid enum member before .value
        cache_provider_val = effective_provider_for_cache.value if isinstance(effective_provider_for_cache, STTProvider) else str(effective_provider_for_cache)
        
        cache_key = self._get_cache_key(audio_hash_part, cache_provider_val, language)
        
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result: logger.debug(f"Cache hit STT key {cache_key[:10]}..."); return cached_result

        _, vad_segments = await self._preprocess_audio(audio_data_bytes)
        
        providers_to_try = self._get_provider_attempt_order(provider_preference)
        if not providers_to_try: raise STTError("No STT providers available.")

        logger.info(f"STT: Order={[p.value for p in providers_to_try]}. Segments={len(vad_segments)}.")

        for current_provider in providers_to_try:
            full_text_parts, confidences = [], []
            provider_succeeded_for_any_segment = False
            try:
                logger.info(f"STT: Attempting provider: {current_provider.value}")
                retryer = AsyncRetrying(
                    stop=stop_after_attempt(settings.MAX_RETRIES), 
                    wait=wait_exponential(multiplier=settings.RETRY_DELAY, min=1, max=5),
                    reraise=False # Handle retry exhaustion for a segment gracefully
                )
                for i, segment in enumerate(vad_segments):
                    if segment.duration_seconds < 0.1: continue # Skip very short segments
                    
                    # `retryer.call` returns None if all attempts fail and reraise=False
                    seg_result_tuple = await retryer.call(self._dispatch_transcribe, current_provider, segment, language)
                    
                    if seg_result_tuple: # If transcription (even empty) was successful for segment
                        text_seg, conf_seg = seg_result_tuple
                        if text_seg: # Only add if text is not empty
                            full_text_parts.append(text_seg)
                            confidences.append(conf_seg)
                            provider_succeeded_for_any_segment = True # Mark success for this provider
                        logger.debug(f"STT: Seg {i+1} by {current_provider.value}: '{text_seg[:30]}...' (Conf: {conf_seg:.2f})")
                    else: # Segment transcription failed after retries
                         logger.warning(f"STT: Seg {i+1} by {current_provider.value} failed all retries.")
                
                if provider_succeeded_for_any_segment and full_text_parts: # Success if at least one segment yielded text
                    final_text = " ".join(full_text_parts).strip()
                    final_conf = float(np.mean(confidences)) if confidences else 0.0
                    if self.cache: self.cache.set(cache_key, (final_text, final_conf), expire=settings.CACHE_TTL, tag=current_provider.value)
                    logger.info(f"STT: Transcribed with {current_provider.value}. Output: '{final_text[:50]}...'")
                    return final_text, final_conf
                elif not provider_succeeded_for_any_segment and vad_segments: # All segments failed for this provider
                     msg = "Provider failed to transcribe any audio segment after retries."
                     logger.warning(f"{current_provider.value}: {msg}")
                     self.provider_errors[current_provider.value] = msg
                # If full_text_parts is empty but provider_succeeded_for_any_segment is false, it means all segments resulted in empty text.
                elif not full_text_parts and vad_segments:
                     msg = "Provider returned no text for any segment."
                     logger.info(f"{current_provider.value}: {msg}") # Info level, as it's not an error, just no speech.
                     self.provider_errors[current_provider.value] = msg # Still record it.

            except ProviderError as e_p: self.provider_errors[e_p.provider] = e_p.message; logger.warning(str(e_p))
            except Exception as e_unexp: logger.error(f"STT: Unexpected error with {current_provider.value}: {e_unexp}", exc_info=True); self.provider_errors[current_provider.value] = f"Unexpected: {str(e_unexp)}"
        
        raise AllProvidersFailedError(self.provider_errors)

    def _get_provider_attempt_order(self, preference: Optional[STTProvider]) -> List[STTProvider]:
        # (Implementation from previous corrected response is fine)
        order: List[STTProvider] = []
        def add_to_order(p: STTProvider):
            if p in self.available_providers and p not in order:
                order.append(p)
        if preference: add_to_order(preference)
        add_to_order(settings.DEFAULT_STT_PROVIDER)
        for fb_p in settings.FALLBACK_STT_PROVIDERS: add_to_order(fb_p)
        for avail_p in self.available_providers: add_to_order(avail_p)
        return order

_stt_client_instance: Optional[MultiSTTClient] = None
def get_multi_stt_client() -> MultiSTTClient:
    global _stt_client_instance
    if _stt_client_instance is None: _stt_client_instance = MultiSTTClient()
    return _stt_client_instance

async def transcribe(audio_data_bytes: bytes, 
                     provider_preference: Optional[STTProvider] = None, 
                     language: Optional[str] = None) -> Tuple[str, float]:
    client = get_multi_stt_client()
    return await client.transcribe(audio_data_bytes, provider_preference, language)