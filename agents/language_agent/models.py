from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class GenerateRequest(BaseModel):
    """Request model for text generation with enhanced capabilities."""
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    template: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    """Enhanced response model for text generation."""
    text: str
    provider: str
    elapsed_time: Optional[float] = None
    template: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    agent: str
    version: str
    timestamp: datetime
    providers: List[str]
    default_provider: str
    templates: List[str]


class AvailableProvidersResponse(BaseModel):
    """Response model for available LLM providers."""
    available: List[str]
    default: str
    fallbacks: List[str]


class TemplateResponse(BaseModel):
    """Response model for available templates."""
    templates: List[str]


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    models: Dict[str, str]
