"""Pydantic models for the Orchestrator Agent."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Request model for the /run endpoint."""

    input: str
    mode: str = "text"
    params: Dict[str, Any] = {}


class StepLog(BaseModel):
    """Model for a single step in the orchestration process."""

    tool: str
    latency_ms: int
    response: Any


class ErrorLog(BaseModel):
    """Model for an error in the orchestration process."""

    tool: str
    message: str


class RunResponse(BaseModel):
    """Response model for the /run endpoint."""

    output: str
    steps: List[StepLog]
    errors: List[ErrorLog]
    audio_url: Optional[str] = None
