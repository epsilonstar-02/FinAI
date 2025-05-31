# Enhanced FastAPI app with multi-provider financial data support

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from .models import (
    PriceRequest, PriceResponse, HistoricalRequest, HistoricalResponse,
    MultiProviderPriceRequest, MultiProviderPriceResponse, ProviderPrice
    # DataProvider enum is now imported from config by models.py
)
from .client import (
    fetch_current_price_with_fallback, fetch_historical_data_with_fallback,
    fetch_current_price_from_specific_provider,
    APIClientError, ProviderNotAvailableError, NoDataAvailableError,
    _financial_data_client # Access client for provider list and specific fetches
)
from .config import settings, DataProvider # Import DataProvider for request models if needed directly

import uvicorn
import asyncio
from datetime import datetime, date
import logging
from typing import List, Optional, Dict, Any
import statistics

# Setup logging (client.py also sets up basicConfig, ensure consistency or centralize)
# logging.basicConfig(level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Enhanced API Agent",
    description="Financial data API with multi-provider support (Alpha Vantage, Yahoo Finance, FMP)",
    version="0.2.1", # Incremented version due to refactoring
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/health", tags=["General"])
def health():
    """Health check endpoint"""
    # Access providers list from the client instance
    providers_list = list(_financial_data_client.providers.keys())
    return {
        "status": "ok", 
        "agent": "API Agent",
        "version": app.version,
        "available_providers": providers_list,
        "timestamp": datetime.utcnow()
    }


@app.get("/providers", tags=["General"])
def get_providers_info():
    """List all available data providers and their configuration"""
    providers_details = {}
    for provider_name, provider_instance in _financial_data_client.providers.items():
        providers_details[provider_name] = {
            "name": provider_name,
            "status": "available", # Assuming if in list, it's configured
            "type": provider_instance.__class__.__name__
        }
        
    return {
        "configured_providers": providers_details,
        "provider_priority_order": settings.PROVIDER_PRIORITY,
        "fallback_enabled": settings.ENABLE_FALLBACK,
        "max_retries": settings.MAX_RETRIES,
        "retry_backoff_factor": settings.RETRY_BACKOFF
    }


@app.get("/price", response_model=PriceResponse, tags=["Data"])
async def get_price(
    symbol: str = Query(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol"),
    provider: Optional[DataProvider] = Query(None, description="Preferred data provider")
):
    """
    Get current price for a symbol.
    Uses a fallback mechanism according to provider priority if preferred provider fails or is not specified.
    """
    try:
        preferred_provider_value = provider.value if provider else None
        data = await fetch_current_price_with_fallback(symbol, preferred_provider_value)
        
        # PriceResponse model expects: symbol, price, timestamp, provider
        # additional_data is Optional and defaults to None by Pydantic if not in data.
        return PriceResponse(**data)
    except APIClientError as e:
        status_code = 502  # Bad Gateway (general upstream failure)
        if isinstance(e, ProviderNotAvailableError):
            status_code = 400  # Bad Request (e.g., requested unavailable preferred provider)
        elif isinstance(e, NoDataAvailableError):
            status_code = 404  # Not Found (no data for symbol from any provider)
        logger.error(f"Error getting price for {symbol}: {str(e)}", exc_info=isinstance(e, APIClientError) and not isinstance(e, (NoDataAvailableError, ProviderNotAvailableError)))
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in /price for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")


@app.get("/historical", response_model=HistoricalResponse, tags=["Data"])
async def get_historical(
    symbol: str = Query(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol"),
    start_date: date = Query(..., description="Start date for historical data (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date for historical data (YYYY-MM-DD)"),
    provider: Optional[DataProvider] = Query(None, description="Preferred data provider")
    # Note: HistoricalRequest.include_volume is in models.py but not used here or in client.
    # Client providers currently always include volume if available.
):
    """
    Get historical OHLCV data for a symbol within a date range.
    Uses a fallback mechanism.
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=400, detail="Start date must be before or same as end date."
        )
    
    # Convert date to datetime for client compatibility if necessary, though client methods accept date objects
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time()) # Use end of day for range inclusivity

    try:
        preferred_provider_value = provider.value if provider else None
        data = await fetch_historical_data_with_fallback(
            symbol, start_dt, end_dt, preferred_provider_value
        )
        
        # HistoricalResponse expects: symbol, timeseries, provider, start_date, end_date
        # metadata is Optional and defaults to None.
        # Client data should contain symbol, timeseries, provider. Add start/end_date for response model.
        response_data = {
            **data,
            "start_date": start_date, # Use original date objects for response
            "end_date": end_date
        }
        return HistoricalResponse(**response_data)
    except APIClientError as e:
        status_code = 502
        if isinstance(e, ProviderNotAvailableError):
            status_code = 400
        elif isinstance(e, NoDataAvailableError):
            status_code = 404
        logger.error(f"Error getting historical for {symbol}: {str(e)}", exc_info=isinstance(e, APIClientError) and not isinstance(e, (NoDataAvailableError, ProviderNotAvailableError)))
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /historical for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")


@app.post("/multi-price", response_model=MultiProviderPriceResponse, tags=["Advanced"])
async def get_multi_provider_price(request: MultiProviderPriceRequest = Body(...)):
    """
    Get current price from multiple specified providers simultaneously.
    Calculates a consensus price if multiple successful responses are received.
    """
    provider_fetch_tasks = []
    
    # Providers in request.providers are DataProvider enum members
    for provider_enum_member in request.providers:
        provider_value = provider_enum_member.value # Get the string value like "yahoo_finance"
        # Use the client method that fetches from a specific provider without fallback
        task = asyncio.create_task(
            _fetch_price_for_multi_provider(request.symbol, provider_value)
        )
        provider_fetch_tasks.append(task)
    
    # Gather results
    price_results: List[ProviderPrice] = await asyncio.gather(*provider_fetch_tasks)
    
    successful_prices = [p.price for p in price_results if p.status == "success" and p.price is not None]
    consensus_price_value: Optional[float] = None
    
    if successful_prices:
        if len(successful_prices) > 1:
            consensus_price_value = statistics.median(successful_prices)
        else: # len == 1
            consensus_price_value = successful_prices[0]
    
    return MultiProviderPriceResponse(
        symbol=request.symbol,
        prices=price_results,
        consensus_price=consensus_price_value,
        timestamp=datetime.utcnow()
    )

async def _fetch_price_for_multi_provider(symbol: str, provider_name: str) -> ProviderPrice:
    """Helper to fetch price from one specific provider for /multi-price endpoint."""
    try:
        # Use client method that targets a specific provider, no internal fallback
        result = await fetch_current_price_from_specific_provider(symbol, provider_name)
        return ProviderPrice(
            provider=provider_name, # or result["provider"] which should be the same
            price=result["price"],
            timestamp=result["timestamp"],
            status="success",
            error_message=None
        )
    except Exception as e:
        logger.warning(f"Error fetching price from '{provider_name}' for '{symbol}' (in /multi-price): {str(e)}")
        return ProviderPrice(
            provider=provider_name,
            price=None, # Model allows Optional float for price now
            timestamp=datetime.utcnow(), # Timestamp of the error/attempt
            status="error",
            error_message=str(e)
        )


@app.get("/compare-providers/{symbol}", tags=["Advanced"])
async def compare_providers(
    symbol: str = Path(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol")
):
    """
    Compare current price data for a symbol across all configured/available providers.
    Provides individual results and summary statistics.
    """
    # Get all currently available providers from the client
    available_providers = list(_financial_data_client.providers.keys())
    comparison_results: Dict[str, Dict[str, Any]] = {}
    tasks = {}
    
    for provider_name in available_providers:
        task = asyncio.create_task(
            _fetch_price_for_comparison(symbol, provider_name)
        )
        tasks[provider_name] = task
    
    for provider_name, task in tasks.items():
        result_data = await task
        comparison_results[provider_name] = result_data
    
    # Calculate statistics from successful fetches
    valid_prices = [
        data["price"] for data in comparison_results.values() 
        if data["status"] == "success" and "price" in data and data["price"] is not None
    ]
    
    stats: Dict[str, Any] = {"count_successful": len(valid_prices)}
    if valid_prices:
        stats["min_price"] = min(valid_prices)
        stats["max_price"] = max(valid_prices)
        stats["mean_price"] = statistics.mean(valid_prices)
        stats["median_price"] = statistics.median(valid_prices) if len(valid_prices) > 0 else None # median needs at least 1
        if len(valid_prices) > 1:
            stats["variance_price"] = statistics.variance(valid_prices)
            stats["stdev_price"] = statistics.stdev(valid_prices)
        else: # len == 1
             stats["median_price"] = valid_prices[0] # Median of 1 is the item itself
             stats["variance_price"] = 0.0
             stats["stdev_price"] = 0.0

    return {
        "symbol": symbol,
        "comparison_timestamp": datetime.utcnow(),
        "provider_results": comparison_results,
        "summary_statistics": stats
    }

async def _fetch_price_for_comparison(symbol: str, provider_name: str) -> Dict[str, Any]:
    """Helper to safely fetch price from a specific provider for /compare-providers."""
    try:
        # Use client method that targets a specific provider, no internal fallback
        data = await fetch_current_price_from_specific_provider(symbol, provider_name)
        # data from client includes: symbol, price, timestamp, provider
        return {"status": "success", **data}
    except Exception as e:
        logger.warning(f"Failed to fetch price from '{provider_name}' for '{symbol}' (in /compare-providers): {str(e)}")
        return {
            "status": "error", 
            "provider": provider_name, # Ensure provider name is in error response
            "error": str(e), 
            "timestamp": datetime.utcnow() # Timestamp of the error
        }

# Main execution for development (uvicorn programmatic run)
if __name__ == "__main__":
    # Recommended: Use `python -m uvicorn agents.api_agent.main:app --host 0.0.0.0 --port 8001 --reload`
    # from command line for better reload and process management.
    uvicorn.run(app, host="0.0.0.0", port=8001) # Removed reload=True for production-like run, manage with CLI.