"""Configuration settings for the Voice Agent with multi-provider support."""
import os
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator, validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class STTProvider(str, Enum):
    """Supported Speech-to-Text providers."""
    WHISPER_LOCAL = "whisper-local"        # OpenAI Whisper (local)
    WHISPER_API = "whisper-api"            # OpenAI Whisper API
    GOOGLE = "google"                      # Google Cloud Speech-to-Text
    AZURE = "azure"                        # Microsoft Azure Speech-to-Text
    VOSK = "vosk"                          # Offline speech recognition
    DEEPSPEECH = "deepspeech"              # Mozilla DeepSpeech (offline)
    SPEECHRECOGNITION = "speechrecognition"  # SpeechRecognition library


class TTSProvider(str, Enum):
    """Supported Text-to-Speech providers."""
    PYTTSX3 = "pyttsx3"                    # Offline TTS engine
    GTTS = "gtts"                          # Google Text-to-Speech
    EDGE = "edge"                          # Microsoft Edge TTS
    ELEVENLABS = "elevenlabs"              # ElevenLabs TTS API
    AMAZON_POLLY = "amazon-polly"          # Amazon Polly TTS
    SILERO = "silero"                      # Silero TTS models (offline)
    COQUI = "coqui"                        # Coqui TTS (offline)


class Settings(BaseSettings):
    """Enhanced settings for the Voice Agent with multi-provider support."""
    
    # Core settings
    LOG_LEVEL: str = "INFO"
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    ENABLE_METRICS: bool = False
    
    # Speech-to-Text settings - Prioritizing free and open-source options
    DEFAULT_STT_PROVIDER: STTProvider = STTProvider.WHISPER_LOCAL  # Free and open-source local model
    FALLBACK_STT_PROVIDERS: List[STTProvider] = [
        STTProvider.SPEECHRECOGNITION,  # Free with offline capability
        STTProvider.VOSK                # Free and open-source
    ]
    
    # Text-to-Speech settings - Prioritizing free and open-source options
    DEFAULT_TTS_PROVIDER: TTSProvider = TTSProvider.PYTTSX3  # Free and offline
    FALLBACK_TTS_PROVIDERS: List[TTSProvider] = [
        TTSProvider.SILERO,  # Free and open-source
        TTSProvider.GTTS     # Free but requires internet
    ]
    
    # Free and Open-Source STT Options
    
    # OpenAI Whisper settings - Free and open-source local model
    WHISPER_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large"] = "base"
    WHISPER_MODEL_PATH: str = "./models/whisper"  # Required for local model
    WHISPER_API_KEY: Optional[str] = None  # Optional paid API service
    
    # Vosk settings - Free and open-source
    VOSK_MODEL_PATH: str = "./models/vosk"
    
    # DeepSpeech settings - Free and open-source
    DEEPSPEECH_MODEL_PATH: str = "./models/deepspeech"
    
    # Free and Open-Source TTS Options
    
    # Silero settings - Free and open-source
    SILERO_MODEL_PATH: str = "./models/silero"
    SILERO_LANGUAGE: str = "en"
    SILERO_SPEAKER: str = "en_0"
    
    # Coqui settings - Free and open-source
    COQUI_MODEL_PATH: str = "./models/coqui"
    
    # NOTE: The following are paid/commercial API services and are OPTIONAL
    # If these API keys are not provided, system will only use free and open-source alternatives
    
    # Google Cloud Speech settings - Paid service
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_LANGUAGE_CODE: str = "en-US"
    
    # Azure Speech settings - Paid service
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    
    # ElevenLabs settings - Paid service
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None
    
    # Amazon Polly settings - Paid service
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    POLLY_VOICE_ID: str = "Joanna"
    
    # Audio processing settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    # Voice Activity Detection settings
    VAD_ENABLED: bool = True
    VAD_AGGRESSIVENESS: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Aggressiveness level for Voice Activity Detection (0-3)",
    )
    
    # Noise reduction settings
    NOISE_REDUCTION_ENABLED: bool = True
    RNNOISE_ENABLED: bool = True
    
    # Cache settings
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 86400  # 24 hours in seconds
    CACHE_DIR: str = "./cache"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
    
    def is_stt_provider_available(self, provider: STTProvider) -> bool:
        """Check if an STT provider is available."""
        if provider == STTProvider.WHISPER_LOCAL:
            return self.WHISPER_MODEL_PATH and Path(self.WHISPER_MODEL_PATH).exists()
        elif provider == STTProvider.WHISPER_API:
            return bool(self.WHISPER_API_KEY)
        elif provider == STTProvider.GOOGLE:
            return bool(self.GOOGLE_APPLICATION_CREDENTIALS) and \
                Path(self.GOOGLE_APPLICATION_CREDENTIALS).exists()
        elif provider == STTProvider.AZURE:
            return bool(self.AZURE_SPEECH_KEY) and bool(self.AZURE_SPEECH_REGION)
        elif provider == STTProvider.VOSK:
            return self.VOSK_MODEL_PATH and Path(self.VOSK_MODEL_PATH).exists()
        elif provider == STTProvider.DEEPSPEECH:
            return self.DEEPSPEECH_MODEL_PATH and Path(self.DEEPSPEECH_MODEL_PATH).exists()
        elif provider == STTProvider.SPEECHRECOGNITION:
            return True  # Always available as it has offline fallback
        return False
    
    def is_tts_provider_available(self, provider: TTSProvider) -> bool:
        """Check if a TTS provider is available."""
        if provider == TTSProvider.PYTTSX3:
            return True  # Always available (offline)
        elif provider == TTSProvider.GTTS:
            return True  # Always available (needs internet)
        elif provider == TTSProvider.EDGE:
            return True  # Always available (needs internet)
        elif provider == TTSProvider.ELEVENLABS:
            return bool(self.ELEVENLABS_API_KEY)
        elif provider == TTSProvider.AMAZON_POLLY:
            return bool(self.AWS_ACCESS_KEY_ID) and bool(self.AWS_SECRET_ACCESS_KEY)
        elif provider == TTSProvider.SILERO:
            return self.SILERO_MODEL_PATH and Path(self.SILERO_MODEL_PATH).exists()
        elif provider == TTSProvider.COQUI:
            return self.COQUI_MODEL_PATH and Path(self.COQUI_MODEL_PATH).exists()
        return False
    
    def get_available_stt_providers(self) -> List[STTProvider]:
        """Get list of available STT providers."""
        return [p for p in STTProvider if self.is_stt_provider_available(p)]
    
    def get_available_tts_providers(self) -> List[TTSProvider]:
        """Get list of available TTS providers."""
        return [p for p in TTSProvider if self.is_tts_provider_available(p)]

    class Config:
        """Pydantic configuration."""

        case_sensitive = True
        extra = "ignore"


# Create a global instance of settings
settings = Settings()
