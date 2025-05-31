# agents/api_agent/main.py
# Adjustments for clarity, error handling, and using the refactored client.

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status # Added status
from fastapi.responses import JSONResponse
from .models import (
    PriceResponse, HistoricalResponse, MultiProviderPriceRequest, 
    MultiProviderPriceResponse, ProviderPrice, OHLC # OHLC needed for type hint if we create it here
)
from .client import ( # Renamed functions and client access
    get_financial_data_client, # Use getter for singleton
    fetch_current_price_with_fallback, fetch_historical_data_with_fallback,
    fetch_current_price_from_specific_provider,
    APIClientError, ProviderNotAvailableError, NoDataAvailableError
)
from .config import settings, DataProvider

import asyncio
from datetime import datetime, date
import logging
from typing import List, Optional, Dict, Any
import statistics

logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Agent", # Simplified title
    description="Financial data API with multi-provider support.",
    version="0.3.0", 
    docs_url="/docs",
    redoc_url="/redoc"
)

# Dependency to get the client instance
def get_client() -> 'FinancialDataClient': # Forward reference FinancialDataClient
    return get_financial_data_client()


@app.on_event("startup")
async def startup_event():
    # Initialize the client on startup to catch config issues early
    get_financial_data_client() 
    logger.info(f"API Agent started. Available providers: {list(get_client().providers.keys())}")


@app.get("/health", tags=["General"])
async def health(client: 'FinancialDataClient' = Depends(get_client)): # Use DI for client
    return {
        "status": "ok", 
        "agent": "API Agent",
        "version": app.version,
        "available_providers": list(client.providers.keys()),
        "timestamp": datetime.utcnow()
    }


@app.get("/providers", tags=["General"])
async def get_providers_info(client: 'FinancialDataClient' = Depends(get_client)):
    providers_details = {}
    for provider_name_str, provider_instance in client.providers.items():
        providers_details[provider_name_str] = {
            "name": provider_name_str,
            "status": "available",
            "type": provider_instance.__class__.__name__
        }
    # Ensure PROVIDER_PRIORITY has string values from enum
    priority_order_str = [dp.value for dp in settings.PROVIDER_PRIORITY]
        
    return {
        "configured_providers": providers_details,
        "provider_priority_order": priority_order_str,
        "fallback_enabled": settings.ENABLE_FALLBACK,
        "max_retries": settings.MAX_RETRIES,
        "retry_backoff_factor": settings.RETRY_BACKOFF
    }


@app.get("/price", response_model=PriceResponse, tags=["Data"])
async def get_price(
    symbol: str = Query(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol"),
    provider: Optional[DataProvider] = Query(None, description="Preferred data provider (enum value like 'yahoo_finance')")
):
    try:
        preferred_provider_str = provider.value if provider else None
        data = await fetch_current_price_with_fallback(symbol, preferred_provider_str)
        return PriceResponse(**data)
    except NoDataAvailableError as e:
        logger.warning(f"No data available for price of {symbol}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ProviderNotAvailableError as e: # e.g. if preferred_provider is not configured
        logger.warning(f"Provider not available for price of {symbol}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except APIClientError as e:
        logger.error(f"API client error getting price for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /price for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected server error occurred.")


@app.get("/historical", response_model=HistoricalResponse, tags=["Data"])
async def get_historical(
    symbol: str = Query(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol"),
    start_date: date = Query(..., description="Start date for historical data (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date for historical data (YYYY-MM-DD)"),
    provider: Optional[DataProvider] = Query(None, description="Preferred data provider (enum value)")
):
    if start_date > end_date:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Start date must be before or same as end date.")
    
    try:
        preferred_provider_str = provider.value if provider else None
        # Client expects date objects directly
        data = await fetch_historical_data_with_fallback(symbol, start_date, end_date, preferred_provider_str)
        # The client now should return start_date and end_date in its dict
        return HistoricalResponse(**data)
    except NoDataAvailableError as e:
        logger.warning(f"No data available for historicals of {symbol}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ProviderNotAvailableError as e:
        logger.warning(f"Provider not available for historicals of {symbol}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except APIClientError as e:
        logger.error(f"API client error getting historicals for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /historical for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected server error occurred.")


async def _fetch_price_for_multi_provider_safe(symbol: str, provider_name_str: str) -> ProviderPrice:
    """Helper to fetch price from one specific provider for /multi-price, safely."""
    try:
        result = await fetch_current_price_from_specific_provider(symbol, provider_name_str)
        return ProviderPrice(
            provider=result["provider"], # Should match provider_name_str
            price=result["price"],
            timestamp=result["timestamp"],
            status="success",
            error_message=None
        )
    except Exception as e: # Catch all errors from specific provider call
        logger.warning(f"Error fetching price from '{provider_name_str}' for '{symbol}' (in /multi-price): {e}")
        return ProviderPrice(
            provider=provider_name_str,
            price=None,
            timestamp=datetime.utcnow(),
            status="error",
            error_message=str(e)
        )

@app.post("/multi-price", response_model=MultiProviderPriceResponse, tags=["Advanced"])
async def get_multi_provider_price(request: MultiProviderPriceRequest = Body(...)):
    # request.providers are DataProvider enum members
    tasks = [
        _fetch_price_for_multi_provider_safe(request.symbol, provider_enum_member.value)
        for provider_enum_member in request.providers
    ]
    
    price_results: List[ProviderPrice] = await asyncio.gather(*tasks)
    
    successful_prices = [p.price for p in price_results if p.status == "success" and p.price is not None]
    consensus_price_value: Optional[float] = None
    
    if successful_prices:
        consensus_price_value = statistics.median(successful_prices) # median handles single item list too
    
    return MultiProviderPriceResponse(
        symbol=request.symbol,
        prices=price_results,
        consensus_price=consensus_price_value,
        timestamp=datetime.utcnow()
    )


async def _fetch_price_for_comparison_safe(symbol: str, provider_name_str: str) -> Dict[str, Any]:
    """Helper to safely fetch price for /compare-providers."""
    try:
        data = await fetch_current_price_from_specific_provider(symbol, provider_name_str)
        return {"status": "success", **data}
    except Exception as e:
        logger.warning(f"Failed to fetch price from '{provider_name_str}' for '{symbol}' (in /compare-providers): {e}")
        return {"status": "error", "provider": provider_name_str, "error": str(e), "timestamp": datetime.utcnow()}

@app.get("/compare-providers/{symbol}", tags=["Advanced"])
async def compare_providers(
    symbol: str = Path(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol"),
    client: 'FinancialDataClient' = Depends(get_client)
):
    available_providers_strs = list(client.providers.keys())
    if not available_providers_strs:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No data providers configured or available.")

    tasks = {
        provider_str: _fetch_price_for_comparison_safe(symbol, provider_str)
        for provider_str in available_providers_strs
    }
    
    # Using asyncio.gather with a dictionary of tasks requires a bit more work to map results
    # Alternative: gather list of tasks and map results back based on original order or provider name in result
    provider_names_ordered = list(tasks.keys())
    gathered_results = await asyncio.gather(*[tasks[name] for name in provider_names_ordered])
    
    comparison_results = {name: result for name, result in zip(provider_names_ordered, gathered_results)}
    
    valid_prices = [
        data["price"] for data in comparison_results.values() 
        if data["status"] == "success" and data.get("price") is not None
    ]
    
    stats: Dict[str, Any] = {"count_successful": len(valid_prices), "count_attempted": len(available_providers_strs)}
    if valid_prices:
        stats["min_price"] = min(valid_prices)
        stats["max_price"] = max(valid_prices)
        stats["mean_price"] = statistics.mean(valid_prices)
        stats["median_price"] = statistics.median(valid_prices)
        if len(valid_prices) > 1:
            stats["stdev_price"] = statistics.stdev(valid_prices)
            stats["variance_price"] = statistics.variance(valid_prices)
        else:
            stats["stdev_price"] = 0.0
            stats["variance_price"] = 0.0

    return {
        "symbol": symbol,
        "comparison_timestamp": datetime.utcnow(),
        "provider_results": comparison_results,
        "summary_statistics": stats
    }


if __name__ == "__main__":
    import uvicorn
    # Corrected uvicorn run command for module structure if main.py is inside api_agent
    # Example: python -m agents.api_agent.main
    uvicorn.run("agents.api_agent.main:app", host="0.0.0.0", port=8001, reload=True)