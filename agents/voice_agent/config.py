"""Configuration settings for the Voice Agent."""
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Settings for the Voice Agent."""

    # Speech-to-Text settings
    STT_PROVIDER: Literal["whisper", "google-stt"] = Field(
        default="whisper",
        description="Provider for speech-to-text conversion",
    )
    MODEL_PATH_STT: str = Field(
        default="./models/whisper",
        description="Path to the Whisper model",
    )

    # Text-to-Speech settings
    TTS_PROVIDER: Literal["gtts", "gemini-tts"] = Field(
        default="gtts",
        description="Provider for text-to-speech conversion",
    )

    # Cache settings
    CACHE_DIR: str = Field(
        default="./cache",
        description="Directory for caching TTS outputs",
    )

    # Voice Activity Detection settings
    VAD_AGGRESSIVENESS: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Aggressiveness level for Voice Activity Detection (0-3)",
    )

    # Noise reduction settings
    RNNOISE_ENABLED: bool = Field(
        default=True,
        description="Whether to apply RNNoise for noise reduction",
    )

    class Config:
        """Pydantic configuration."""

        case_sensitive = True
        extra = "ignore"


# Create a global instance of settings
settings = Settings()
