from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from jinja2 import Environment, FileSystemLoader
import os
import asyncio
from pathlib import Path

from .models import GenerateRequest, GenerateResponse
from .llm_client import generate_text, LLMClientError
from .config import settings

# Create FastAPI app
app = FastAPI(title="Language Agent", version="0.1.0")

# Set up Jinja2 environment
templates_path = Path(__file__).parent / "prompts"
env = Environment(loader=FileSystemLoader(templates_path))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "agent": "Language Agent"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text based on query and context using a template.
    
    Args:
        request: The generation request containing query and context
        
    Returns:
        GenerateResponse with the generated text
        
    Raises:
        HTTPException: If there's an error in text generation
    """
    try:
        # Load the template for market brief
        template = env.get_template("market_brief.tpl")
        
        # Render the template with the request data
        prompt = template.render(
            query=request.query,
            context=request.context
        )
        
        # Generate text using the prompt
        generated_text = await generate_text(prompt)
        
        # Return the response
        return GenerateResponse(text=generated_text)
    
    except LLMClientError as e:
        # Map LLMClientError to HTTP 502 Bad Gateway
        raise HTTPException(status_code=502, detail=str(e))
    
    except ValidationError as e:
        # Handle validation errors
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Import app for uvicorn
app = app  # This makes the app importable for uvicorn
