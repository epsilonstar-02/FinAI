# Enhanced FastAPI app with multi-provider financial data support

from fastapi import FastAPI, HTTPException, Depends, Query, Body, Path
from fastapi.responses import JSONResponse
from .models import (
    PriceRequest, PriceResponse, HistoricalRequest, HistoricalResponse,
    MultiProviderPriceRequest, MultiProviderPriceResponse, ProviderPrice,
    DataProvider
)
from .client import (
    fetch_current_price, fetch_historical, APIClientError, 
    FinancialDataClient, _client,
    ProviderNotAvailableError, NoDataAvailableError
)
from .config import settings

import uvicorn
import asyncio
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Enhanced API Agent",
    description="Financial data API with multi-provider support (Alpha Vantage, Yahoo Finance, FMP)",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/health")
def health():
    """Health check endpoint"""
    providers = list(_client.providers.keys())
    return {
        "status": "ok", 
        "agent": "API Agent",
        "version": "0.2.0",
        "available_providers": providers,
        "timestamp": datetime.utcnow()
    }


@app.get("/providers")
def get_providers():
    """List all available data providers"""
    providers = {}
    for provider_name, provider in _client.providers.items():
        providers[provider_name] = {
            "name": provider_name,
            "available": True,
            "type": provider.__class__.__name__
        }
        
    return {
        "providers": providers,
        "default_priority": settings.PROVIDER_PRIORITY,
        "fallback_enabled": settings.ENABLE_FALLBACK
    }


@app.get("/price", response_model=PriceResponse)
async def get_price(request: PriceRequest = Depends()):
    """Get current price for a symbol with optional provider preference"""
    try:
        # Pass the preferred provider if specified
        preferred_provider = request.provider.value if request.provider else None
        data = await fetch_current_price(request.symbol, preferred_provider)
        
        # If additional_data not in response, add it as None for schema compatibility
        if "additional_data" not in data:
            data["additional_data"] = None
            
        return data
    except APIClientError as e:
        error_msg = str(e)
        status_code = 502
        
        if isinstance(e, ProviderNotAvailableError):
            status_code = 400  # Bad Request - provider not available
        elif isinstance(e, NoDataAvailableError):
            status_code = 404  # Not Found - no data for symbol
            
        raise HTTPException(status_code=status_code, detail=error_msg)


@app.get("/historical", response_model=HistoricalResponse)
async def get_historical(request: HistoricalRequest = Depends()):
    """Get historical data for a symbol with optional provider preference"""
    if request.start > request.end:
        raise HTTPException(
            status_code=400, detail="start date must be before end date"
        )
    
    try:
        # Pass the preferred provider if specified
        preferred_provider = request.provider.value if request.provider else None
        data = await fetch_historical(
            request.symbol, request.start, request.end, preferred_provider
        )
        
        # Add required fields for enhanced schema
        if "provider" not in data:
            data["provider"] = "default"
        if "start_date" not in data:
            data["start_date"] = request.start
        if "end_date" not in data:
            data["end_date"] = request.end
        if "metadata" not in data:
            data["metadata"] = None
            
        return data
    except APIClientError as e:
        error_msg = str(e)
        status_code = 502
        
        if isinstance(e, ProviderNotAvailableError):
            status_code = 400  # Bad Request - provider not available
        elif isinstance(e, NoDataAvailableError):
            status_code = 404  # Not Found - no data for symbol
            
        raise HTTPException(status_code=status_code, detail=error_msg)


@app.post("/multi-price", response_model=MultiProviderPriceResponse)
async def get_multi_provider_price(request: MultiProviderPriceRequest):
    """Get price from multiple providers simultaneously and calculate consensus"""
    prices = []
    tasks = []
    
    # Create async tasks for each provider
    for provider in request.providers:
        task = asyncio.create_task(_fetch_provider_price(request.symbol, provider.value))
        tasks.append((provider.value, task))
    
    # Wait for all tasks to complete
    for provider_name, task in tasks:
        price = await task
        prices.append(price)
    
    # Calculate consensus price if we have successful results
    successful_prices = [p for p in prices if p.status == "success"]
    consensus_price = None
    
    if successful_prices:
        price_values = [p.price for p in successful_prices]
        consensus_price = statistics.median(price_values) if len(price_values) > 1 else price_values[0]
    
    return {
        "symbol": request.symbol,
        "prices": prices,
        "consensus_price": consensus_price,
        "timestamp": datetime.utcnow()
    }


async def _fetch_provider_price(symbol: str, provider: str) -> ProviderPrice:
    """Helper function to fetch price from a specific provider with error handling"""
    try:
        result = await _client.fetch_current_price(symbol, provider)
        return ProviderPrice(
            provider=provider,
            price=result["price"],
            timestamp=result["timestamp"],
            status="success",
            error_message=None
        )
    except Exception as e:
        logger.error(f"Error fetching price from {provider}: {str(e)}")
        return ProviderPrice(
            provider=provider,
            price=0.0,
            timestamp=datetime.utcnow(),
            status="error",
            error_message=str(e)
        )


@app.get("/compare/{symbol}")
async def compare_providers(symbol: str):
    """Compare data across all available providers for a symbol"""
    providers = list(_client.providers.keys())
    results = {}
    tasks = {}
    
    # Create tasks for each provider
    for provider in providers:
        task = asyncio.create_task(_safe_fetch_price(symbol, provider))
        tasks[provider] = task
    
    # Wait for all tasks to complete
    for provider, task in tasks.items():
        result = await task
        results[provider] = result
    
    # Calculate statistics if we have enough data
    valid_prices = [data["price"] for provider, data in results.items() 
                   if data["status"] == "success" and "price" in data]
    
    stats = {}
    if valid_prices:
        stats["count"] = len(valid_prices)
        stats["min"] = min(valid_prices)
        stats["max"] = max(valid_prices)
        stats["mean"] = statistics.mean(valid_prices) if valid_prices else None
        stats["median"] = statistics.median(valid_prices) if len(valid_prices) > 1 else valid_prices[0] if valid_prices else None
        stats["variance"] = statistics.variance(valid_prices) if len(valid_prices) > 1 else 0
    
    return {
        "symbol": symbol,
        "providers": results,
        "statistics": stats,
        "timestamp": datetime.utcnow()
    }


async def _safe_fetch_price(symbol: str, provider: str) -> Dict[str, Any]:
    """Safely fetch price data from a provider"""
    try:
        data = await _client.fetch_current_price(symbol, provider)
        return {"status": "success", **data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run("agents.api_agent.main:app", host="0.0.0.0", port=8001, reload=True)
