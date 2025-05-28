from fastapi import FastAPI, HTTPException, Depends, Query, Request, status, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from jinja2 import Environment, FileSystemLoader
import os
import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from .models import (
    GenerateRequest, 
    GenerateResponse, 
    HealthResponse, 
    AvailableProvidersResponse,
    TemplateResponse,
    ModelInfoResponse
)
from .multi_llm_client import get_multi_llm_client, generate_text, LLMClientError, AllProvidersFailedError, ProviderError
from .config import settings, LLMProvider

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Language Agent", 
    description="Multi-provider language model agent for FinAI",
    version="0.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 environment
templates_path = Path(__file__).parent / settings.TEMPLATES_DIR
env = Environment(loader=FileSystemLoader(templates_path))

# Initialize the multi-LLM client
llm_client = get_multi_llm_client()

# Rate limiting settings
REQUEST_COUNTS = {}
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS = 100        # requests per duration

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean up old entries
    for ip in list(REQUEST_COUNTS.keys()):
        if current_time - REQUEST_COUNTS[ip]["timestamp"] > RATE_LIMIT_DURATION:
            del REQUEST_COUNTS[ip]
    
    # Check if client has exceeded rate limit
    if client_ip in REQUEST_COUNTS:
        request_info = REQUEST_COUNTS[client_ip]
        if current_time - request_info["timestamp"] < RATE_LIMIT_DURATION:
            if request_info["count"] >= MAX_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please try again later."}
                )
            request_info["count"] += 1
        else:
            # Reset if outside the window
            REQUEST_COUNTS[client_ip] = {"count": 1, "timestamp": current_time}
    else:
        # First request from this IP
        REQUEST_COUNTS[client_ip] = {"count": 1, "timestamp": current_time}
    
    # Process the request
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Get available providers
    available_providers = settings.get_available_providers()
    available_providers_str = [p.value for p in available_providers]
    
    return HealthResponse(
        status="ok",
        agent="Language Agent",
        version="0.2.0",
        timestamp=datetime.utcnow(),
        providers=available_providers_str,
        default_provider=settings.DEFAULT_PROVIDER.value,
        templates=list(map(lambda x: x.stem, templates_path.glob("*.tpl")))
    )


@app.get("/providers", response_model=AvailableProvidersResponse)
async def get_available_providers():
    """Get available LLM providers"""
    available_providers = settings.get_available_providers()
    return AvailableProvidersResponse(
        available=[p.value for p in available_providers],
        default=settings.DEFAULT_PROVIDER.value,
        fallbacks=[p.value for p in settings.FALLBACK_PROVIDERS if p in available_providers]
    )


@app.get("/templates", response_model=TemplateResponse)
async def get_templates():
    """Get available templates"""
    templates = list(map(lambda x: x.stem, templates_path.glob("*.tpl")))
    return TemplateResponse(templates=templates)


@app.get("/models", response_model=ModelInfoResponse)
async def get_models():
    """Get information about available models"""
    model_info = {
        LLMProvider.GOOGLE.value: settings.GEMINI_MODEL,
        LLMProvider.OPENAI.value: settings.OPENAI_MODEL,
        LLMProvider.ANTHROPIC.value: settings.ANTHROPIC_MODEL,
        LLMProvider.HUGGINGFACE.value: settings.HUGGINGFACE_MODEL,
        LLMProvider.LLAMA.value: settings.LLAMA_MODEL_PATH,
    }
    
    # Filter to only include available providers
    available_providers = settings.get_available_providers()
    available_models = {p.value: model_info[p.value] for p in available_providers if p.value in model_info}
    
    return ModelInfoResponse(models=available_models)


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    provider: Optional[str] = Query(None, description="Optional specific LLM provider to use")
):
    """
    Generate text based on query and context using a template.
    
    Args:
        request: The generation request containing query and context
        provider: Optional specific LLM provider to use
        
    Returns:
        GenerateResponse with the generated text
        
    Raises:
        HTTPException: If there's an error in text generation
    """
    start_time = time.time()
    
    try:
        # Validate provider if specified
        provider_enum = None
        if provider:
            try:
                provider_enum = LLMProvider(provider)
                if provider_enum not in settings.get_available_providers():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Provider '{provider}' is not available. Available providers: {[p.value for p in settings.get_available_providers()]}"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider '{provider}'. Valid providers: {[p.value for p in LLMProvider]}"
                )
        
        # Get template name from request or use default
        template_name = f"{request.template}.tpl" if request.template else "market_brief.tpl"
        
        try:
            # Load the template
            template = env.get_template(template_name)
        except Exception as e:
            # If template not found, return error
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_name}' not found: {str(e)}"
            )
        
        # Render the template with the request data
        prompt = template.render(
            query=request.query,
            context=request.context
        )
        
        # Generate text using the prompt
        generated_text = await generate_text(prompt, provider_enum)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return the response
        return GenerateResponse(
            text=generated_text,
            provider=provider if provider else settings.DEFAULT_PROVIDER.value,
            elapsed_time=elapsed_time,
            template=request.template or "market_brief"
        )
    
    except AllProvidersFailedError as e:
        # Return detailed error when all providers fail
        raise HTTPException(
            status_code=502,
            detail={
                "message": "All LLM providers failed",
                "provider_errors": e.provider_errors
            }
        )
    
    except ProviderError as e:
        # Return error for specific provider
        raise HTTPException(
            status_code=502,
            detail=f"Provider error ({e.provider}): {e.message}"
        )
    
    except LLMClientError as e:
        # Map LLMClientError to HTTP 502 Bad Gateway
        raise HTTPException(status_code=502, detail=str(e))
    
    except ValidationError as e:
        # Handle validation errors
        raise HTTPException(status_code=422, detail=str(e))
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Import app for uvicorn
app = app  # This makes the app importable for uvicorn
