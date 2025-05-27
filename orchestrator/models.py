"""Pydantic models for the Orchestrator Agent."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Request model for the /run endpoint."""

    input: str
    mode: Literal["text"] = "text"


class RunStep(BaseModel):
    """Model for a single step in the orchestration process."""

    tool: str
    response: Any


class RunResponse(BaseModel):
    """Response model for the /run endpoint."""

    output: str
    steps: List[RunStep]
