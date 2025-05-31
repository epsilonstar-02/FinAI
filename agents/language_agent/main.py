# agents/language_agent/main.py
# Enhanced error handling, startup checks, and consistency.

from fastapi import (
    FastAPI, HTTPException, Depends, Query, Request, status, Body, Path as FastApiPath
) # Added FastApiPath for clarity
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError # Explicitly import for catching
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import os
import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import (
    GenerateRequest, GenerateResponse, HealthResponse, 
    AvailableProvidersResponse, TemplateResponse, ModelInfoResponse
)
from .multi_llm_client import (
    get_multi_llm_client, generate_text_from_client as generate_text, # Use aliased public function
    LLMClientError, AllProvidersFailedError, ProviderError
)
from .config import settings, LLMProvider

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Language Agent", 
    description="Multi-provider language model agent for FinAI",
    version="0.3.1" # Incremented for this refactor
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Jinja2 Environment
_jinja_env: Optional[Environment] = None

def get_jinja_env() -> Environment:
    global _jinja_env
    if _jinja_env is None:
        # Resolve templates_path relative to this main.py file's parent directory
        module_dir = Path(__file__).resolve().parent
        templates_dir_path = module_dir / settings.TEMPLATES_DIR
        if not templates_dir_path.is_dir():
            logger.error(f"Templates directory '{templates_dir_path}' not found or not a directory.")
            # This is a critical setup error. Application might not function correctly.
            # We could raise an error here to prevent startup, or let it fail on first template access.
            # For now, Jinja2 will raise TemplateNotFound later.
        _jinja_env = Environment(loader=FileSystemLoader(str(templates_dir_path)), autoescape=True)
        logger.info(f"Jinja2 environment initialized. Templates path: {templates_dir_path}")
    return _jinja_env

# LLM Client (singleton)
_llm_client_instance: Optional['MultiLLMClient'] = None # Forward reference

def get_client_dependency() -> 'MultiLLMClient': # Renamed for clarity as a dependency getter
    global _llm_client_instance
    if _llm_client_instance is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="LLM Client is not initialized. Service is unavailable.")
    return _llm_client_instance

# Rate limiting (simple in-memory)
_REQUEST_COUNTS: Dict[str, List[float]] = {} # Store timestamps of requests
_RATE_LIMIT_WINDOW_SECONDS = 60
_MAX_REQUESTS_PER_WINDOW = 100 # Example limit

@app.middleware("http")
async def rate_limit_middleware_refined(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown_ip"
    current_time = time.time()
    
    # Get or initialize timestamps for this IP
    ip_timestamps = _REQUEST_COUNTS.get(client_ip, [])
    
    # Filter out timestamps older than the window
    ip_timestamps = [ts for ts in ip_timestamps if current_time - ts < _RATE_LIMIT_WINDOW_SECONDS]
    
    if len(ip_timestamps) >= _MAX_REQUESTS_PER_WINDOW:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            content={"detail": "Too many requests. Please try again later."})
    
    ip_timestamps.append(current_time)
    _REQUEST_COUNTS[client_ip] = ip_timestamps
    
    response = await call_next(request)
    return response


@app.on_event("startup")
async def startup_event():
    global _llm_client_instance
    try:
        # Initialize Jinja env first (less likely to fail critically than LLM client)
        get_jinja_env() # This will initialize _jinja_env
        _llm_client_instance = get_multi_llm_client() # Initialize client
        logger.info(f"Application startup complete. Default provider: {settings.DEFAULT_PROVIDER.value}. "
                    f"Available: {[p.value for p in _llm_client_instance.available_providers]}")
    except Exception as e:
        logger.critical(f"LLM Client failed to initialize during startup: {e}", exc_info=True)
        _llm_client_instance = None # Ensure it's None if init fails


@app.exception_handler(AllProvidersFailedError)
async def all_providers_failed_handler(request: Request, exc: AllProvidersFailedError):
    logger.error(f"All LLM providers failed for request {request.url.path}: {exc.provider_errors}", exc_info=False) # No need for stack trace here
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=jsonable_encoder({"detail": "All LLM providers failed to generate a response.", "errors": exc.provider_errors})
    )

@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    logger.error(f"LLM Provider error for {request.url.path} ({exc.provider}): {exc.message}", exc_info=False)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=jsonable_encoder({"detail": f"LLM Provider error ({exc.provider}).", "message": exc.message})
    )

@app.exception_handler(LLMClientError)
async def llm_client_error_handler(request: Request, exc: LLMClientError):
    logger.error(f"LLM Client Error for {request.url.path}: {exc}", exc_info=True) # Stack trace might be useful
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=jsonable_encoder({"detail": "LLM service is currently unavailable or misconfigured.", "message": str(exc)})
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    logger.warning(f"Request validation error for {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": "Request validation failed.", "errors": exc.errors()})
    )

@app.exception_handler(TemplateNotFound)
async def template_not_found_handler(request: Request, exc: TemplateNotFound):
    logger.error(f"Template not found for {request.url.path}: {exc.name}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=jsonable_encoder({"detail": f"Template '{exc.name}' not found."})
    )

@app.exception_handler(Exception) # Generic fallback
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder({"detail": "An unexpected internal server error occurred.", "message": str(exc)})
    )


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check_endpoint(client: 'MultiLLMClient' = Depends(get_client_dependency)): # Use new dep name
    template_files: List[str] = []
    jinja_env = get_jinja_env() # Ensure env is initialized
    try:
        # list_templates() might not work with all loaders, globbing is more reliable for FileSystemLoader
        module_dir = Path(__file__).resolve().parent
        templates_dir_path = module_dir / settings.TEMPLATES_DIR
        if templates_dir_path.is_dir():
            template_files = sorted([p.stem for p in templates_dir_path.glob("*.tpl")])
    except Exception as e:
        logger.warning(f"Could not list templates during health check: {e}")

    return HealthResponse(
        status="ok" if client else "degraded", # Client might be None if startup failed
        agent="Language Agent",
        version=app.version,
        timestamp=datetime.utcnow(),
        providers=[p.value for p in client.available_providers] if client else [],
        default_provider=settings.DEFAULT_PROVIDER.value if client else "N/A (Client Init Failed)",
        templates=template_files
    )

@app.get("/providers", response_model=AvailableProvidersResponse, tags=["Utility"])
async def get_available_providers_endpoint(client: 'MultiLLMClient' = Depends(get_client_dependency)):
    available_fallbacks = [
        p.value for p in settings.FALLBACK_PROVIDERS if p in client.available_providers
    ]
    return AvailableProvidersResponse(
        available=[p.value for p in client.available_providers],
        default=settings.DEFAULT_PROVIDER.value,
        fallbacks=available_fallbacks
    )

@app.get("/templates", response_model=TemplateResponse, tags=["Utility"])
async def get_templates_list_endpoint(): # No client needed
    template_files: List[str] = []
    jinja_env = get_jinja_env()
    try:
        module_dir = Path(__file__).resolve().parent
        templates_dir_path = module_dir / settings.TEMPLATES_DIR
        if templates_dir_path.is_dir():
            template_files = sorted([p.stem for p in templates_dir_path.glob("*.tpl")])
        else: # Should have been caught by Jinja init log, but good to check
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Templates directory not configured or found.")
    except Exception as e:
        logger.error(f"Error listing templates: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not list templates.")
    return TemplateResponse(templates=template_files)

@app.get("/models", response_model=ModelInfoResponse, tags=["Utility"])
async def get_models_info_endpoint(client: 'MultiLLMClient' = Depends(get_client_dependency)):
    model_info_map = {
        LLMProvider.GOOGLE: settings.GEMINI_MODEL,
        LLMProvider.OPENAI: settings.OPENAI_MODEL,
        LLMProvider.ANTHROPIC: settings.ANTHROPIC_MODEL,
        LLMProvider.HUGGINGFACE: settings.HUGGINGFACE_MODEL,
        LLMProvider.LLAMA: settings.LLAMA_MODEL_PATH,
    }
    available_models_dict = {
        p.value: model_info_map.get(p, "N/A") # Use .get for safety
        for p in client.available_providers 
    }
    return ModelInfoResponse(models=available_models_dict)


@app.post("/generate", response_model=GenerateResponse, tags=["Core"])
async def generate_endpoint( # Renamed from `generate` to avoid conflict with imported function
    request: GenerateRequest = Body(...),
    client: 'MultiLLMClient' = Depends(get_client_dependency), # Inject client
    # provider query param is string, will be converted to LLMProvider enum
    provider_query: Optional[str] = Query(None, alias="provider", description=f"Optional: specific LLM provider (e.g., {', '.join([p.value for p in LLMProvider])})")
):
    start_time_ns = time.perf_counter_ns() # More precise timing
    
    provider_enum_preference: Optional[LLMProvider] = None
    if provider_query:
        try:
            provider_enum_preference = LLMProvider(provider_query.lower())
            if provider_enum_preference not in client.available_providers:
                valid_options_str = ", ".join([p.value for p in client.available_providers]) or "None"
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Requested provider '{provider_query}' is not available. Available: [{valid_options_str}]"
                )
        except ValueError: # Invalid enum value
            valid_enums_str = ", ".join([p.value for p in LLMProvider])
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider name '{provider_query}'. Valid names: [{valid_enums_str}]"
            )
    
    template_name_str = request.template if request.template else "market_brief"
    template_file_name = f"{template_name_str}.tpl"
    jinja_env = get_jinja_env()
    
    try:
        template = jinja_env.get_template(template_file_name)
    except TemplateNotFound: # Jinja2's specific exception
        logger.warning(f"Template '{template_file_name}' not found in {jinja_env.loader.searchpath}.")
        raise # Will be caught by template_not_found_handler
    except Exception as e_tpl: # Other potential Jinja errors
        logger.error(f"Error loading template '{template_file_name}': {e_tpl}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not load template '{template_name_str}'.")
        
    try:
        prompt = template.render(query=request.query, context=request.context or {})
    except Exception as e_render: # Errors during template rendering
        logger.error(f"Error rendering template '{template_name_str}': {e_render}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error rendering template '{template_name_str}'. Check context and template syntax.")

    # Prepare per-request generation arguments, ensuring None values are not passed if not set
    generation_args: Dict[str, Any] = {}
    if request.max_tokens is not None: generation_args["max_tokens"] = request.max_tokens
    if request.temperature is not None: generation_args["temperature"] = request.temperature
    
    # generate_text is the imported public function from multi_llm_client
    generated_text_content, used_provider_enum = await generate_text(
        prompt, 
        provider=provider_enum_preference, 
        generation_args=generation_args if generation_args else None
    )
    
    elapsed_time_s = (time.perf_counter_ns() - start_time_ns) / 1_000_000_000 # Seconds
    
    return GenerateResponse(
        text=generated_text_content,
        provider=used_provider_enum.value,
        elapsed_time=round(elapsed_time_s, 4),
        template=template_name_str
    )


if __name__ == "__main__":
    import uvicorn
    # For running directly: python -m agents.language_agent.main
    uvicorn.run("agents.language_agent.main:app", host="0.0.0.0", port=8005, reload=True, log_level=settings.LOG_LEVEL.lower())