# agents/voice_agent/config.py
# No significant changes anticipated, mostly ensuring LOG_LEVEL is applied and paths are clear.
# Original content preserved with minor adjustments.

import os
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
import logging # Added

from dotenv import load_dotenv
from pydantic import Field, field_validator # validator removed, using field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict # Added SettingsConfigDict

load_dotenv()

class STTProvider(str, Enum):
    WHISPER_LOCAL = "whisper-local"
    WHISPER_API = "whisper-api"
    GOOGLE = "google"
    AZURE = "azure"
    VOSK = "vosk"
    DEEPSPEECH = "deepspeech"
    SPEECHRECOGNITION = "speechrecognition"

class TTSProvider(str, Enum):
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    EDGE = "edge"
    ELEVENLABS = "elevenlabs"
    AMAZON_POLLY = "amazon-polly"
    SILERO = "silero"
    COQUI = "coqui"

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    TIMEOUT: int = 30 # General timeout for external API calls
    MAX_RETRIES: int = 2 # Default retries for provider calls
    RETRY_DELAY: int = 1 # Base delay in seconds for retry backoff
    ENABLE_METRICS: bool = False # Placeholder for potential future metrics

    DEFAULT_STT_PROVIDER: STTProvider = STTProvider.WHISPER_LOCAL
    FALLBACK_STT_PROVIDERS: List[STTProvider] = [STTProvider.SPEECHRECOGNITION, STTProvider.VOSK]
    
    DEFAULT_TTS_PROVIDER: TTSProvider = TTSProvider.PYTTSX3
    FALLBACK_TTS_PROVIDERS: List[TTSProvider] = [TTSProvider.SILERO, TTSProvider.GTTS]
    
    # Whisper settings
    WHISPER_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large"] = "base"
    # Ensure model paths are resolvable. Using absolute paths or paths relative to a defined project root is best.
    # For now, assuming relative to where the app runs or a mapped volume in Docker.
    WHISPER_MODEL_PATH: str = os.getenv("WHISPER_MODEL_PATH", "./models/whisper") 
    WHISPER_API_KEY: Optional[str] = None
    
    # Vosk settings
    VOSK_MODEL_PATH: str = os.getenv("VOSK_MODEL_PATH", "./models/vosk")
    
    # DeepSpeech settings
    DEEPSPEECH_MODEL_PATH: str = os.getenv("DEEPSPEECH_MODEL_PATH", "./models/deepspeech")
    
    # Silero settings
    SILERO_MODEL_PATH: str = os.getenv("SILERO_MODEL_PATH", "./models/silero") # Path to custom .pt model if any
    SILERO_LANGUAGE: str = "en"
    SILERO_SPEAKER: str = "en_0" # Default Silero speaker ID
    
    # Coqui settings
    COQUI_MODEL_PATH: str = os.getenv("COQUI_MODEL_PATH", "./models/coqui") # Path to custom Coqui model if any
    COQUI_DEFAULT_MODEL_NAME: str = "tts_models/en/ljspeech/tacotron2-DDC" # Default if path not found

    # Google Cloud Speech settings
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_LANGUAGE_CODE: str = "en-US"
    
    # Azure Speech settings
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    
    # ElevenLabs settings
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = "Rachel" # Example default voice ID
    
    # Amazon Polly settings
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1" # Default AWS region for Polly
    POLLY_VOICE_ID: str = "Joanna" # Example default Polly voice
    
    # Audio processing settings
    SAMPLE_RATE: int = 16000 # Standard for many STT/TTS
    CHANNELS: int = 1 # Mono
    
    VAD_ENABLED: bool = True
    VAD_AGGRESSIVENESS: int = Field(default=2, ge=0, le=3)
    
    NOISE_REDUCTION_ENABLED: bool = True
    RNNOISE_ENABLED: bool = True # Specific flag for RNNoise, if available
    
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 86400 # 24 hours
    CACHE_DIR: str = "./cache/voice_agent" # Specific cache dir for this agent

    model_config = SettingsConfigDict( # Pydantic V2 style
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # case_sensitive defaults to False in Pydantic v2 BaseSettings,
        # explicitly set if True is required by original logic (matching env var case)
        # case_sensitive = True 
    )

    # ---- Helper methods for provider availability ----
    def _check_path_exists(self, path_str: Optional[str], is_dir: bool = True) -> bool:
        if not path_str: return False
        p = Path(path_str).resolve() # Resolve to absolute path
        return p.is_dir() if is_dir else p.is_file()

    def is_stt_provider_available(self, provider: STTProvider) -> bool:
        if provider == STTProvider.WHISPER_LOCAL:
            # For local Whisper, model files are downloaded by the library to WHISPER_MODEL_PATH.
            # The path itself existing is a good first check. Actual model files determined by size.
            return self._check_path_exists(self.WHISPER_MODEL_PATH, is_dir=True)
        elif provider == STTProvider.WHISPER_API:
            return bool(self.WHISPER_API_KEY)
        elif provider == STTProvider.GOOGLE:
            return self._check_path_exists(self.GOOGLE_APPLICATION_CREDENTIALS, is_dir=False)
        elif provider == STTProvider.AZURE:
            return bool(self.AZURE_SPEECH_KEY and self.AZURE_SPEECH_REGION)
        elif provider == STTProvider.VOSK:
            return self._check_path_exists(self.VOSK_MODEL_PATH, is_dir=True)
        elif provider == STTProvider.DEEPSPEECH:
            # DeepSpeech model usually consists of a .pbmm model file and optionally a .scorer file
            # Checking for a common model file pattern might be better. For now, check dir.
            return self._check_path_exists(self.DEEPSPEECH_MODEL_PATH, is_dir=True)
        elif provider == STTProvider.SPEECHRECOGNITION:
            return True # Library itself is "available", actual engines might vary.
        return False
    
    def is_tts_provider_available(self, provider: TTSProvider) -> bool:
        if provider == TTSProvider.PYTTSX3: return True
        elif provider == TTSProvider.GTTS: return True
        elif provider == TTSProvider.EDGE: return True
        elif provider == TTSProvider.ELEVENLABS: return bool(self.ELEVENLABS_API_KEY)
        elif provider == TTSProvider.AMAZON_POLLY: return bool(self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY)
        elif provider == TTSProvider.SILERO:
            # Silero can download models or use a path. If path is set, check it.
            # Otherwise, assume it can download if needed.
            return True # Simplified: assume available, will try to download if path not set/found
        elif provider == TTSProvider.COQUI:
            # Coqui can download models. If path set, check it.
            return True # Simplified: assume available
        return False
    
    def get_available_stt_providers(self) -> List[STTProvider]:
        return [p for p in STTProvider if self.is_stt_provider_available(p)]
    
    def get_available_tts_providers(self) -> List[TTSProvider]:
        return [p for p in TTSProvider if self.is_tts_provider_available(p)]

settings = Settings()

# Ensure cache directory exists
Path(settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)
# Ensure local model directories exist if paths are specified, as some libs expect them.
# This is a light check; actual model files are the real dependency.
for model_path_attr in ["WHISPER_MODEL_PATH", "VOSK_MODEL_PATH", "DEEPSPEECH_MODEL_PATH", "SILERO_MODEL_PATH", "COQUI_MODEL_PATH"]:
    path_val = getattr(settings, model_path_attr, None)
    if path_val:
        Path(path_val).mkdir(parents=True, exist_ok=True)


# Configure logging
log_level_to_set = settings.LOG_LEVEL.upper()
if not hasattr(logging, log_level_to_set):
    logging.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in Voice Agent settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s (VOICE_AGENT) - %(levelname)s - %(message)s"
)
# Re-get logger after basicConfig
logger = logging.getLogger(__name__)