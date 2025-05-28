"""Pydantic schemas for the Voice Agent API with multi-provider support."""
from typing import Optional, Dict, List, Any, Union
from datetime import datetime

from fastapi import UploadFile
from pydantic import BaseModel, Field, validator


class STTRequest(BaseModel):
    """Schema for Speech-to-Text request."""
    file: UploadFile = Field(..., description="Audio file to transcribe")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en-US')")


class STTResponse(BaseModel):
    """Schema for Speech-to-Text response with provider information."""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score for the transcription")
    provider: str = Field(..., description="STT provider used for transcription")
    elapsed_time: Optional[float] = Field(None, description="Time taken for transcription in seconds")
    language: Optional[str] = Field(None, description="Detected or specified language")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Segments with timestamps if available")


class TTSRequest(BaseModel):
    """Schema for Text-to-Speech request with enhanced options."""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field("default", description="Voice to use for synthesis")
    speed: float = Field(1.0, description="Speed factor for speech (1.0 is normal)")
    pitch: Optional[float] = Field(None, description="Pitch adjustment (provider dependent)")
    volume: Optional[float] = Field(None, description="Volume adjustment (provider dependent)")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en-US')")
    ssml: bool = Field(False, description="Whether the text is SSML")

    @validator('speed')
    def validate_speed(cls, v):
        """Validate speed is within reasonable bounds."""
        if v <= 0.0 or v > 3.0:
            raise ValueError("Speed must be between 0.0 and 3.0")
        return v


class TTSResponse(BaseModel):
    """Schema for Text-to-Speech response."""
    audio: bytes = Field(..., description="Audio data as bytes")
    provider: str = Field(..., description="TTS provider used for synthesis")
    format: str = Field("mp3", description="Audio format")
    duration: Optional[float] = Field(None, description="Duration of audio in seconds")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Status of the service")
    agent: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    timestamp: datetime = Field(..., description="Current timestamp")
    stt_providers: List[str] = Field(..., description="Available STT providers")
    tts_providers: List[str] = Field(..., description="Available TTS providers")
    default_stt_provider: str = Field(..., description="Default STT provider")
    default_tts_provider: str = Field(..., description="Default TTS provider")


class ProviderInfo(BaseModel):
    """Schema for provider information."""
    available: List[str] = Field(..., description="Available providers")
    default: str = Field(..., description="Default provider")
    fallbacks: List[str] = Field(..., description="Fallback providers")


class AvailableProvidersResponse(BaseModel):
    """Schema for available providers response."""
    stt: ProviderInfo = Field(..., description="STT provider information")
    tts: ProviderInfo = Field(..., description="TTS provider information")


class VoiceListResponse(BaseModel):
    """Schema for voice list response."""
    provider: str = Field(..., description="Provider name or 'all'")
    voices: Optional[List[str]] = Field(None, description="List of available voices for a specific provider")
    voices_by_provider: Optional[Dict[str, List[str]]] = Field(None, description="Voices grouped by provider")

    @validator('voices', 'voices_by_provider')
    def validate_voices(cls, v, values):
        """Validate that either voices or voices_by_provider is provided."""
        provider = values.get('provider')
        if provider == 'all' and v is None and values.get('voices_by_provider') is None:
            raise ValueError("For 'all' provider, voices_by_provider must be provided")
        if provider != 'all' and values.get('voices') is None:
            raise ValueError("For specific provider, voices must be provided")
        return v
