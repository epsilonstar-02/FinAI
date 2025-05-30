"""Main module for the Analysis Agent."""
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import Dict, Any, Callable, Optional, List, Union
import logging
import time
from functools import lru_cache

# Internal imports
from agents.analysis_agent.config import settings
from agents.analysis_agent.calculator import build_summary
from agents.analysis_agent.models import (
    AnalyzeRequest, 
    AnalyzeResponse, 
    HistoricalDataPoint,
    ProviderInfo,
    ErrorResponse,
    RiskMetrics
)
from agents.analysis_agent.providers import get_provider, AnalysisProvider

# Configure logging - now done in config.py
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Analysis Agent",
    description="Financial analysis microservice for the FinAI platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Create an in-memory cache for analysis results
if settings.CACHE_ENABLED:
    analysis_cache = {}
    
# Rate limiting state
if settings.RATE_LIMIT_ENABLED:
    rate_limit_state = {"count": 0, "reset_time": time.time() + 60}
    
# Provider instance cache
@lru_cache(maxsize=10)
def get_provider_instance(provider_name: str) -> AnalysisProvider:
    """Get a cached provider instance."""
    return get_provider(provider_name)

def create_error_response(
    status_code: int, message: str, details: Any = None
) -> JSONResponse:
    """Create a standardized error response."""
    error_info = {
        "status": "error",
        "message": message,
    }
    if details is not None:
        error_info["details"] = str(details)
    
    return JSONResponse(
        status_code=status_code,
        content=error_info,
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler to catch all unhandled exceptions."""
    logger.exception("Unhandled exception occurred: %s", exc) # Added %s for better logging
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred",
        details=str(exc),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Validation error",
        details=exc.errors(),
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return create_error_response(
        status_code=exc.status_code,
        message=exc.detail,
    )

@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {"status": "ok", "agent": "Analysis Agent"}

@app.post("/analyze", response_model=AnalyzeResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze financial data and generate insights.
    
    Args:
        request: Analysis request with prices and historical data
        
    Returns:
        Analysis response with exposures, changes, volatility and summary
        
    Raises:
        HTTPException: If there's an error during analysis
    """
    # Apply rate limiting if enabled
    if settings.RATE_LIMIT_ENABLED:
        current_time = time.time()
        if current_time > rate_limit_state["reset_time"]:
            rate_limit_state["count"] = 0
            rate_limit_state["reset_time"] = current_time + 60
            
        rate_limit_state["count"] += 1
        if rate_limit_state["count"] > settings.RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
    
    # Check cache if enabled
    if settings.CACHE_ENABLED:
        # Simple cache key based on request parameters
        cache_key = f"{request.provider}_{str(request.prices)}_{str(request.historical)}"
        if cache_key in analysis_cache:
            cache_entry = analysis_cache[cache_key]
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] < settings.CACHE_TTL:
                logger.info(f"Cache hit for analysis request")
                return cache_entry["response"]
    
    # Use specified provider or default from settings
    provider_name = request.provider or settings.ANALYSIS_PROVIDER
    start_time = time.time()
    
    # Create provider info object
    provider_info = ProviderInfo(
        name=provider_name,
        version="1.0.0",
        fallback_used=False
    )
    
    try:
        # Get provider instance
        provider = get_provider_instance(provider_name)
        
        # Compute financial metrics
        exposures = provider.compute_exposures(request.prices)
        changes = provider.compute_changes(request.historical)
        volatility = provider.compute_volatility(
            request.historical, 
            window=settings.VOLATILITY_WINDOW
        )
        
        # Optional calculations based on request and settings
        correlations = None
        risk_metrics = None
        
        if request.include_correlations and settings.ENABLE_CORRELATION_ANALYSIS:
            correlations = provider.compute_correlations(request.historical)
            
        if request.include_risk_metrics and settings.ENABLE_RISK_METRICS:
            risk_metrics = provider.compute_risk_metrics(request.historical)
        
        # Generate summary with alerts
        summary = build_summary(
            exposures, 
            changes, 
            volatility, 
            threshold=settings.ALERT_THRESHOLD
        )
        
    except Exception as e:
        logger.exception(f"Error with provider {provider_name}: {str(e)}")
        
        # Try fallback providers if available
        for fallback_name in settings.FALLBACK_PROVIDERS:
            if fallback_name == provider_name:
                continue  # Skip if it's the same as the failed provider
                
            try:
                logger.info(f"Attempting fallback to provider: {fallback_name}")
                fallback_provider = get_provider_instance(fallback_name)
                
                # Compute metrics with fallback
                exposures = fallback_provider.compute_exposures(request.prices)
                changes = fallback_provider.compute_changes(request.historical)
                volatility = fallback_provider.compute_volatility(
                    request.historical, 
                    window=settings.VOLATILITY_WINDOW
                )
                
                # Optional calculations
                correlations = None
                risk_metrics = None
                
                if request.include_correlations and settings.ENABLE_CORRELATION_ANALYSIS:
                    correlations = fallback_provider.compute_correlations(request.historical)
                    
                if request.include_risk_metrics and settings.ENABLE_RISK_METRICS:
                    risk_metrics = fallback_provider.compute_risk_metrics(request.historical)
                
                # Generate summary
                summary = build_summary(
                    exposures, 
                    changes, 
                    volatility, 
                    threshold=settings.ALERT_THRESHOLD
                )
                
                # Update provider info to indicate fallback was used
                provider_info = ProviderInfo(
                    name=fallback_name,
                    version="1.0.0",
                    fallback_used=True
                )
                
                # Success with fallback
                break
                
            except Exception as fallback_error:
                logger.exception(f"Fallback provider {fallback_name} also failed: {str(fallback_error)}")
                continue
        else:
            # If we get here, all providers failed
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed with all providers. Original error: {str(e)}"
            )
    
    # Calculate execution time
    execution_time_ms = (time.time() - start_time) * 1000
    provider_info.execution_time_ms = execution_time_ms
    
    # Create response
    response = AnalyzeResponse(
        exposures=exposures,
        changes=changes,
        volatility=volatility,
        correlations=correlations,
        risk_metrics=risk_metrics,
        summary=summary,
        provider_info=provider_info
    )
    
    # Update cache if enabled
    if settings.CACHE_ENABLED:
        cache_key = f"{request.provider}_{str(request.prices)}_{str(request.historical)}"
        analysis_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Simple cache size management - remove oldest entries if too many
        if len(analysis_cache) > 1000:  # Arbitrary limit
            oldest_key = min(analysis_cache.keys(), key=lambda k: analysis_cache[k]["timestamp"])
            del analysis_cache[oldest_key]
    
    return response