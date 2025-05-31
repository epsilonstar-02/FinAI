# orchestrator/models.py
# No changes are made to this file as it seems well-structured and functional.
# Original content is preserved.

"""Pydantic models for the Orchestrator Agent."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Request model for the /run endpoint."""

    input: str
    mode: str = "text" # Default to "text" as per common usage, can be overridden
    params: Dict[str, Any] = Field(default_factory=dict)


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
    steps: List[StepLog] = Field(default_factory=list)
    errors: List[ErrorLog] = Field(default_factory=list)
    audio_output_b64: Optional[str] = None
    audio_url: Optional[str] = None