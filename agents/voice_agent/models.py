# agents/voice_agent/models.py
# No significant changes needed. Models are well-defined.
# Added default_factory for timestamp in HealthResponse.
# Ensured Pydantic v2 field_validator syntax.

from typing import Optional, Dict, List, Any
from datetime import datetime

from fastapi import UploadFile # Retain for potential direct use, though STTRequest uses it
from pydantic import BaseModel, Field, field_validator # validator removed

class STTRequest(BaseModel): # This model might not be used if STT endpoint takes File directly
    file: Any # Placeholder, FastAPI uses UploadFile directly in endpoint
    language: Optional[str] = Field(None, description="Language code (e.g., 'en-US')")

class STTResponse(BaseModel):
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0), if available")
    provider: str = Field(..., description="STT provider used for transcription")
    elapsed_time: Optional[float] = Field(None, description="Time taken in seconds")
    language: Optional[str] = Field(None, description="Detected or specified language")
    # segments: Optional[List[Dict[str, Any]]] = Field(None, description="Segments with timestamps if available") # Keep if VAD/segmentation provides this


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: Optional[str] = Field("default", description="Voice ID or name for synthesis")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speed factor (0.25x to 4x)") # Adjusted bounds
    # pitch: Optional[float] = Field(None, description="Pitch adjustment (provider dependent)") # Keep if used by any provider
    # volume: Optional[float] = Field(None, description="Volume adjustment (provider dependent)") # Keep if used
    language: Optional[str] = Field(None, description="Language code (e.g., 'en-US')")
    # ssml: bool = Field(False, description="Whether the text is SSML") # Keep if used


class TTSResponse(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio data (if not streaming)")
    provider: str = Field(..., description="TTS provider used")
    format: str = Field("mp3", description="Audio format (e.g., mp3, wav)")
    # duration: Optional[float] = Field(None, description="Duration of audio in seconds") # Can be hard to get reliably pre-decode
    elapsed_time: Optional[float] = Field(None, description="Time taken for synthesis in seconds")


class HealthResponse(BaseModel):
    status: str
    agent: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stt_providers: List[str]
    tts_providers: List[str]
    default_stt_provider: str
    default_tts_provider: str

class ProviderInfo(BaseModel):
    available: List[str]
    default: str
    fallbacks: List[str]

class AvailableProvidersResponse(BaseModel):
    stt: ProviderInfo
    tts: ProviderInfo

class VoiceListResponse(BaseModel):
    provider: str
    voices: Optional[List[str]] = None
    voices_by_provider: Optional[Dict[str, List[str]]] = None

    @field_validator('voices_by_provider', mode='before') # Check logic, might need model_validator
    @classmethod
    def check_voices_structure(cls, v, info): # Pydantic v2: info.data to access other fields
        values = info.data
        is_all_providers = values.get('provider') == 'all'
        has_voices = values.get('voices') is not None
        has_voices_by_provider = v is not None

        if is_all_providers and not has_voices_by_provider:
            raise ValueError("For 'all' provider, 'voices_by_provider' must be provided.")
        if not is_all_providers and not has_voices:
            raise ValueError("For a specific provider, 'voices' list must be provided.")
        if has_voices and has_voices_by_provider:
            raise ValueError("Provide either 'voices' or 'voices_by_provider', not both.")
        return v