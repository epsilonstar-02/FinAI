# agents/language_agent/models.py
# No significant changes needed, models are well-defined.
# Added default_factory for timestamp.

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class GenerateRequest(BaseModel):
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    template: Optional[str] = None # Base name of the template file, e.g., "market_brief"
    max_tokens: Optional[int] = Field(None, ge=1, description="Override default max tokens.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Override default temperature.")
    # Can add top_p, top_k here later if needed for request-specific overrides.


class GenerateResponse(BaseModel):
    text: str
    provider: str 
    elapsed_time: Optional[float] = None # In seconds
    template: Optional[str] = None
    # Consider adding token counts if providers return them (prompt_tokens, completion_tokens)


class HealthResponse(BaseModel):
    status: str
    agent: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    providers: List[str] 
    default_provider: str
    templates: List[str]


class AvailableProvidersResponse(BaseModel):
    available: List[str]
    default: str
    fallbacks: List[str] # List of actually available fallback providers


class TemplateResponse(BaseModel):
    templates: List[str] # List of template base names (without .tpl extension)


class ModelInfoResponse(BaseModel):
    models: Dict[str, str] # provider_name -> model_identifier_string