from pydantic import BaseModel
from typing import Dict, Any


class GenerateRequest(BaseModel):
    """
    Request model for text generation.
    
    Attributes:
        query: The user query string
        context: Dictionary containing context information for generation
    """
    query: str
    context: Dict[str, Any]


class GenerateResponse(BaseModel):
    """
    Response model for text generation.
    
    Attributes:
        text: The generated text
    """
    text: str
