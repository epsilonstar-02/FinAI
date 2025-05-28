"""Pydantic schemas for the Voice Agent API."""
from typing import Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field


class STTRequest(BaseModel):
    """Schema for Speech-to-Text request."""

    file: UploadFile = Field(..., description="Audio file to transcribe")


class STTResponse(BaseModel):
    """Schema for Speech-to-Text response."""

    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score for the transcription")


class TTSRequest(BaseModel):
    """Schema for Text-to-Speech request."""

    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice to use for synthesis")
    speed: float = Field(default=1.0, description="Speed factor for speech")


class TTSResponse(BaseModel):
    """Schema for Text-to-Speech response."""

    audio: bytes = Field(..., description="Audio data as bytes or base64 encoded string")
