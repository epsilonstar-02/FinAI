"""Main module for the Analysis Agent."""
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Dict, Any, Callable
import logging

from agents.analysis_agent.config import settings
from agents.analysis_agent.models import AnalyzeRequest, AnalyzeResponse, HistoricalDataPoint
from agents.analysis_agent.calculator import (
    compute_exposures,
    compute_changes,
    compute_volatility,
    build_summary,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Analysis Agent",
    description="Financial analysis microservice for the FinAI platform",
    version="0.1.0",
)

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

@app.post("/analyze", response_model=AnalyzeResponse)
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
    # Compute financial metrics
    exposures = compute_exposures(request.prices)
    changes = compute_changes(request.historical)
    volatility = compute_volatility(
        request.historical, 
        window=settings.VOLATILITY_WINDOW
    )
    
    # Generate summary with alerts
    summary = build_summary(
        exposures, 
        changes, 
        volatility, 
        threshold=settings.ALERT_THRESHOLD
    )
    
    # Return response
    return AnalyzeResponse(
        exposures=exposures,
        changes=changes,
        volatility=volatility,
        summary=summary
    )