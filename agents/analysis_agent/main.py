# agents/analysis_agent/main.py
# Refined caching key, error handling, and provider interaction.

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import Dict, Any, Optional, List
import logging
import time
import hashlib # For robust cache key
import json # For robust cache key

from agents.analysis_agent.config import settings
from agents.analysis_agent.calculator import build_summary # build_summary is a presentation concern
from agents.analysis_agent.models import (
    AnalyzeRequest, AnalyzeResponse, HistoricalDataPoint, # HistoricalDataPoint used for casting
    ProviderInfo, ErrorResponse, RiskMetrics
)
from agents.analysis_agent.providers import get_provider, AnalysisProvider

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Analysis Agent",
    description="Financial analysis microservice for the FinAI platform",
    version="0.2.0", # Updated version
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# In-memory cache for analysis results (can be replaced with Redis/Memcached for production)
_analysis_cache: Dict[str, Dict[str, Any]] = {} # Cache key -> {"response_data": ..., "timestamp": ...}

# Rate limiting state (simple in-memory)
_rate_limit_counts: Dict[str, List[float]] = {} # client_ip -> list of request timestamps
_RATE_LIMIT_WINDOW_SECONDS = 60

def _generate_cache_key(request: AnalyzeRequest) -> str:
    """Generates a more robust cache key for an AnalyzeRequest."""
    # Exclude provider from cache key if it's a fallback scenario,
    # but for direct requests, provider choice matters.
    # model_dump for consistent serialization.
    # Sorting dicts within historical data is important if order doesn't matter for analysis outcome.
    # HistoricalDataPoint itself is a dict, its order of keys is stable.
    # The list of HistoricalDataPoint for each symbol should be sorted by date before hashing
    # if the analysis is invariant to original list order but dependent on chronological order.
    # For simplicity now, direct dump.
    dumped_request = request.model_dump(exclude_none=True) # exclude_none to make cache key more stable
    # Provider is part of the request model, so it's included in dump.
    
    # Normalize historical data for caching: sort by date within each symbol's list
    if 'historical' in dumped_request and dumped_request['historical']:
        for symbol_hist in dumped_request['historical'].values():
            if isinstance(symbol_hist, list):
                # Sort based on 'date' field of each HistoricalDataPoint dictionary
                try:
                    symbol_hist.sort(key=lambda x: x.get('date', ''))
                except Exception as e_sort_cache:
                    logger.warning(f"Could not sort historical data for cache key generation: {e_sort_cache}")


    try:
        serialized_request = json.dumps(dumped_request, sort_keys=True)
        return hashlib.sha256(serialized_request.encode('utf-8')).hexdigest()
    except TypeError as e_ser: # Should not happen with model_dump if models are JSON serializable
        logger.error(f"Failed to serialize request for cache key: {e_ser}. Using fallback key.")
        # Fallback to simpler key (less robust)
        return f"{request.provider}_{str(request.prices)}_{str(request.historical)}"


@app.middleware("http")
async def rate_limit_and_time_middleware(request: Request, call_next):
    # Rate Limiting
    if settings.RATE_LIMIT_ENABLED:
        client_ip = request.client.host if request.client else "unknown_ip"
        current_time = time.time()
        
        ip_timestamps = _rate_limit_counts.get(client_ip, [])
        ip_timestamps = [ts for ts in ip_timestamps if current_time - ts < _RATE_LIMIT_WINDOW_SECONDS] # Filter old
        
        if len(ip_timestamps) >= settings.RATE_LIMIT:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                content={"detail": "Rate limit exceeded. Please try again later."})
        ip_timestamps.append(current_time)
        _rate_limit_counts[client_ip] = ip_timestamps
    
    # Process Timing
    start_time_req = time.time()
    response = await call_next(request)
    process_time_req = time.time() - start_time_req
    response.headers["X-Process-Time-Seconds"] = str(round(process_time_req, 4))
    return response

# Centralized error response creation
def _create_error_json_response(status_code: int, message: str, details: Optional[Any] = None) -> JSONResponse:
    error_content = {"status": "error", "message": message}
    if details:
        try: # Ensure details are serializable
            json.dumps(details) 
            error_content["details"] = details
        except TypeError:
            error_content["details"] = str(details) # Fallback to string
    return JSONResponse(status_code=status_code, content=error_content)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler_custom(request: Request, exc: RequestValidationError):
    return _create_error_json_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "Validation error", exc.errors())

@app.exception_handler(HTTPException) # Handles HTTPExceptions raised by our code
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    return _create_error_json_response(exc.status_code, exc.detail) # Pass detail as message

@app.exception_handler(Exception) # Generic fallback
async def global_exception_handler_custom(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url.path}: {exc}", exc_info=True)
    return _create_error_json_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "An unexpected internal error occurred", str(exc))


@app.get("/health", tags=["Utility"])
async def health_check_endpoint() -> Dict[str, str]:
    return {"status": "ok", "agent": "Analysis Agent", "version": app.version}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"],
          summary="Analyze financial data",
          description="Performs financial analysis including exposures, changes, volatility, correlations, and risk metrics based on input price and historical data.")
async def analyze_endpoint(request_data: AnalyzeRequest) -> AnalyzeResponse: # Renamed for clarity
    # request_data.historical contains List[Dict], needs to be List[HistoricalDataPoint] for providers
    # Pydantic should handle this conversion if request_data is type hinted as AnalyzeRequest.
    # However, let's ensure the historical data is correctly formed for internal use.
    historical_typed: Dict[str, List[HistoricalDataPoint]] = {
        symbol: [HistoricalDataPoint(**dp) for dp in dp_list]
        for symbol, dp_list in request_data.historical.items()
    } if request_data.historical else {}


    if settings.CACHE_ENABLED:
        cache_key = _generate_cache_key(request_data) # Use robust key generation
        if cache_key in _analysis_cache:
            cached_item = _analysis_cache[cache_key]
            if time.time() - cached_item["timestamp"] < settings.CACHE_TTL:
                logger.info(f"Cache hit for analysis request (key: {cache_key[:10]}...).")
                # Ensure response is AnalyzeResponse model
                return AnalyzeResponse(**cached_item["response_data"]) 
            else:
                logger.info(f"Cache stale for key {cache_key[:10]}..., removing.")
                del _analysis_cache[cache_key] # Stale entry

    # Actual analysis logic
    provider_name_to_use = request_data.provider or settings.ANALYSIS_PROVIDER
    analysis_start_time = time.time()
    
    provider_info = ProviderInfo(name=provider_name_to_use, version="1.0.0") # Version can be dynamic
    
    analysis_succeeded = False
    response_payload_dict: Optional[Dict[str, Any]] = None # Store result as dict first

    # Attempt with primary provider then fallbacks
    providers_to_try = [provider_name_to_use] + [fp for fp in settings.FALLBACK_PROVIDERS if fp != provider_name_to_use]

    for current_provider_name in providers_to_try:
        try:
            logger.info(f"Attempting analysis with provider: {current_provider_name}")
            provider_instance = get_provider(current_provider_name) # Cached provider instance

            exposures = provider_instance.compute_exposures(request_data.prices)
            changes = provider_instance.compute_changes(historical_typed)
            volatility = provider_instance.compute_volatility(historical_typed, settings.VOLATILITY_WINDOW)
            
            correlations_result: Optional[Dict[str, Dict[str, float]]] = None
            if request_data.include_correlations and settings.ENABLE_CORRELATION_ANALYSIS:
                correlations_result = provider_instance.compute_correlations(historical_typed)
            
            risk_metrics_result: Optional[Dict[str, Optional[RiskMetrics]]] = None
            if request_data.include_risk_metrics and settings.ENABLE_RISK_METRICS:
                risk_metrics_result = provider_instance.compute_risk_metrics(historical_typed)
            
            summary_text = build_summary(exposures, changes, volatility, settings.ALERT_THRESHOLD)

            execution_time_ms = (time.time() - analysis_start_time) * 1000
            provider_info.name = current_provider_name # Update to actual used provider
            provider_info.execution_time_ms = round(execution_time_ms, 2)
            if current_provider_name != provider_name_to_use: # If we used a fallback
                provider_info.fallback_used = True
            
            response_payload_dict = {
                "exposures": exposures, "changes": changes, "volatility": volatility,
                "correlations": correlations_result, "risk_metrics": risk_metrics_result,
                "summary": summary_text, "provider_info": provider_info.model_dump()
            }
            analysis_succeeded = True
            break # Success, exit loop

        except Exception as e_provider:
            logger.error(f"Analysis failed with provider '{current_provider_name}': {e_provider}", exc_info=True)
            if current_provider_name == providers_to_try[-1]: # Last provider also failed
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Analysis failed with all providers. Last error ({current_provider_name}): {e_provider}")
            # else, loop continues to next fallback provider
    
    if not analysis_succeeded or response_payload_dict is None: # Should be caught by loop end exception
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis could not be completed.")

    # Convert dict to AnalyzeResponse model instance for validation and serialization
    final_response = AnalyzeResponse(**response_payload_dict)

    if settings.CACHE_ENABLED:
        cache_key_to_store = _generate_cache_key(request_data) # Regenerate key just in case (though should be same)
        _analysis_cache[cache_key_to_store] = {
            "response_data": final_response.model_dump(), # Store as dict
            "timestamp": time.time()
        }
        # Simple cache eviction if too large
        if len(_analysis_cache) > 1000: # Example limit
            oldest_key = min(_analysis_cache, key=lambda k: _analysis_cache[k]["timestamp"])
            del _analysis_cache[oldest_key]
            logger.info(f"Cache limit reached, removed oldest entry: {oldest_key[:10]}...")
    
    return final_response

# Example for OpenAPI schema (if not using schema_extra in models)
# Moved from models.py to keep models cleaner.
if not app.openapi_schema:
    from fastapi.openapi.utils import get_openapi
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Add example for AnalyzeRequest to openapi schema
    request_example = {
        "prices": {"AAPL": 150.25, "MSFT": 245.80},
        "historical": {
            "AAPL": [{"date": "2023-05-01", "close": 150.25, "open": 150.0, "high": 151.0, "low": 149.0, "volume": 1000000},
                       {"date": "2023-04-30", "close": 149.50, "open": 149.0, "high": 150.0, "low": 148.5, "volume": 900000}],
            "MSFT": [{"date": "2023-05-01", "close": 245.80}, {"date": "2023-04-30", "close": 245.00}]
        },
        "provider": "advanced",
        "include_correlations": True,
        "include_risk_metrics": True
    }
    # This is a bit manual; Pydantic schema_extra is cleaner.
    # For /analyze endpoint, find its path definition and add example
    # This part is usually better handled via Pydantic's `schema_extra` in the model itself.
    # If keeping it here, ensure it targets the correct schema component.
    # For simplicity, will rely on Pydantic's schema_extra in models.py if that was intended.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agents.analysis_agent.main:app", host=settings.HOST, port=settings.PORT, reload=True, log_level=settings.LOG_LEVEL.lower())