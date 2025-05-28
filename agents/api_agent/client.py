# Enhanced API client with multiple open-source financial data providers and fallback mechanisms

import httpx
from .config import settings, DataProvider
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from alpha_vantage.timeseries import TimeSeries

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Base exception for API client errors"""
    pass


class ProviderNotAvailableError(APIClientError):
    """Raised when a specific data provider is not available"""
    pass


class NoDataAvailableError(APIClientError):
    """Raised when no data is available for the requested symbol"""
    pass


class FinancialDataProvider:
    """Base class for financial data providers"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol"""
        raise NotImplementedError
        
    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get historical data for a symbol between start and end dates"""
        raise NotImplementedError


class YahooFinanceProvider(FinancialDataProvider):
    """Yahoo Finance data provider (open-source)"""
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price using yfinance (Yahoo Finance)"""
        try:
            # Execute in a separate thread to not block the event loop
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            data = await loop.run_in_executor(None, lambda: ticker.history(period="1d"))
            
            if data.empty:
                raise NoDataAvailableError(f"No Yahoo Finance data for symbol {symbol}")
                
            price = data['Close'].iloc[-1]
            timestamp = datetime.utcnow()
            return {"symbol": symbol, "price": float(price), "timestamp": timestamp, "provider": "yahoo_finance"}
        except Exception as e:
            logger.error(f"Yahoo Finance error: {str(e)}")
            raise APIClientError(f"Yahoo Finance error: {str(e)}")
    
    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get historical data using yfinance (Yahoo Finance)"""
        try:
            # Execute in a separate thread to not block the event loop
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            data = await loop.run_in_executor(
                None, 
                lambda: ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            )
            
            if data.empty:
                raise NoDataAvailableError(f"No Yahoo Finance historical data for {symbol}")
                
            timeseries = []
            for date, row in data.iterrows():
                timeseries.append({
                    "date": date.date(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
            
            timeseries.sort(key=lambda x: x["date"])
            return {"symbol": symbol, "timeseries": timeseries, "provider": "yahoo_finance"}
        except Exception as e:
            logger.error(f"Yahoo Finance historical data error: {str(e)}")
            raise APIClientError(f"Yahoo Finance error: {str(e)}")


class AlphaVantageProvider(FinancialDataProvider):
    """Alpha Vantage data provider with enhanced error handling"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 5):
        super().__init__(timeout)
        self.api_key = api_key
        self.base_url = base_url
        self.client = TimeSeries(key=api_key, output_format='json')
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price using Alpha Vantage API"""
        try:
            # Make the API call in a non-blocking way
            loop = asyncio.get_event_loop()
            data, _ = await loop.run_in_executor(None, lambda: self.client.get_quote_endpoint(symbol=symbol))
            
            if not data:
                raise NoDataAvailableError(f"No Alpha Vantage data for {symbol}")
                
            price = float(data.get("05. price", 0))
            timestamp = datetime.utcnow()
            return {"symbol": symbol, "price": price, "timestamp": timestamp, "provider": "alpha_vantage"}
        except Exception as e:
            logger.error(f"Alpha Vantage error: {str(e)}")
            raise APIClientError(f"Alpha Vantage error: {str(e)}")
    
    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get historical data using Alpha Vantage API"""
        try:
            # Make the API call in a non-blocking way
            loop = asyncio.get_event_loop()
            data, meta_data = await loop.run_in_executor(
                None, lambda: self.client.get_daily_adjusted(symbol=symbol, outputsize='full')
            )
            
            if not data:
                raise NoDataAvailableError(f"No Alpha Vantage historical data for {symbol}")
                
            timeseries = []
            for date_str, daily in data.items():
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start.date() <= dt <= end.date():
                    timeseries.append({
                        "date": dt,
                        "open": float(daily["1. open"]),
                        "high": float(daily["2. high"]),
                        "low": float(daily["3. low"]),
                        "close": float(daily["4. close"]),
                        "volume": int(daily["6. volume"])
                    })
            
            timeseries.sort(key=lambda x: x["date"])
            return {"symbol": symbol, "timeseries": timeseries, "provider": "alpha_vantage"}
        except Exception as e:
            logger.error(f"Alpha Vantage historical data error: {str(e)}")
            raise APIClientError(f"Alpha Vantage error: {str(e)}")


class FMPProvider(FinancialDataProvider):
    """Financial Modeling Prep API provider (free tier)"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 5):
        super().__init__(timeout)
        self.api_key = api_key
        self.base_url = base_url
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price using Financial Modeling Prep API"""
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise APIClientError(f"FMP API returned {response.status_code}")
                
            data = response.json()
            if not data or len(data) == 0:
                raise NoDataAvailableError(f"No FMP data for {symbol}")
                
            quote = data[0]  # FMP returns an array
            price = float(quote.get("price", 0))
            timestamp = datetime.utcnow()
            return {"symbol": symbol, "price": price, "timestamp": timestamp, "provider": "financial_modeling_prep"}
        except Exception as e:
            logger.error(f"FMP error: {str(e)}")
            raise APIClientError(f"FMP error: {str(e)}")
    
    async def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get historical data using Financial Modeling Prep API"""
        try:
            url = f"{self.base_url}/historical-price-full/{symbol}"
            params = {
                "apikey": self.api_key,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d")
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise APIClientError(f"FMP API returned {response.status_code}")
                
            data = response.json()
            if not data or "historical" not in data or not data["historical"]:
                raise NoDataAvailableError(f"No FMP historical data for {symbol}")
                
            timeseries = []
            for item in data["historical"]:
                dt = datetime.strptime(item["date"], "%Y-%m-%d").date()
                timeseries.append({
                    "date": dt,
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": int(item["volume"])
                })
            
            timeseries.sort(key=lambda x: x["date"])
            return {"symbol": symbol, "timeseries": timeseries, "provider": "financial_modeling_prep"}
        except Exception as e:
            logger.error(f"FMP historical data error: {str(e)}")
            raise APIClientError(f"FMP error: {str(e)}")


class FinancialDataClient:
    """Multi-provider financial data client with fallback mechanisms"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured data providers"""
        # Initialize Yahoo Finance (no API key needed)
        self.providers[DataProvider.YAHOO_FINANCE] = YahooFinanceProvider(timeout=settings.TIMEOUT)
        
        # Initialize Alpha Vantage if API key is available
        if settings.ALPHA_VANTAGE_KEY:
            self.providers[DataProvider.ALPHA_VANTAGE] = AlphaVantageProvider(
                api_key=settings.ALPHA_VANTAGE_KEY,
                base_url=settings.ALPHA_VANTAGE_URL,
                timeout=settings.TIMEOUT
            )
        
        # Initialize FMP if API key is available
        if settings.FMP_KEY:
            self.providers[DataProvider.FMP] = FMPProvider(
                api_key=settings.FMP_KEY,
                base_url=settings.FMP_URL,
                timeout=settings.TIMEOUT
            )
        
        logger.info(f"Initialized {len(self.providers)} financial data providers")
    
    async def _try_provider(self, provider_name: str, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Try to get data from a specific provider"""
        if provider_name not in self.providers:
            raise ProviderNotAvailableError(f"Provider {provider_name} not available")
            
        provider = self.providers[provider_name]
        method = getattr(provider, method_name)
        return await method(*args, **kwargs)
    
    @retry(
        retry=retry_if_exception_type(APIClientError),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_BACKOFF)
    )
    async def fetch_current_price(self, symbol: str, preferred_provider: Optional[str] = None) -> Dict[str, Any]:
        """Fetch current price with fallback to alternative providers"""
        errors = []
        
        # Use preferred provider if specified, otherwise use priority list
        providers_to_try = [preferred_provider] if preferred_provider else settings.PROVIDER_PRIORITY
        
        # Try each provider in order
        for provider in providers_to_try:
            if provider not in self.providers:
                continue
                
            try:
                result = await self._try_provider(provider, "get_current_price", symbol)
                logger.info(f"Got price for {symbol} from {provider}")
                return result
            except APIClientError as e:
                errors.append({"provider": provider, "error": str(e)})
                if not settings.ENABLE_FALLBACK:
                    break
        
        # If we got here, all providers failed
        error_details = "\n".join([f"{e['provider']}: {e['error']}" for e in errors])
        raise APIClientError(f"All providers failed to get price for {symbol}:\n{error_details}")
    
    @retry(
        retry=retry_if_exception_type(APIClientError),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_BACKOFF)
    )
    async def fetch_historical(self, symbol: str, start: datetime, end: datetime, 
                           preferred_provider: Optional[str] = None) -> Dict[str, Any]:
        """Fetch historical data with fallback to alternative providers"""
        errors = []
        
        # Use preferred provider if specified, otherwise use priority list
        providers_to_try = [preferred_provider] if preferred_provider else settings.PROVIDER_PRIORITY
        
        # Try each provider in order
        for provider in providers_to_try:
            if provider not in self.providers:
                continue
                
            try:
                result = await self._try_provider(provider, "get_historical_data", symbol, start, end)
                logger.info(f"Got historical data for {symbol} from {provider}")
                return result
            except APIClientError as e:
                errors.append({"provider": provider, "error": str(e)})
                if not settings.ENABLE_FALLBACK:
                    break
        
        # If we got here, all providers failed
        error_details = "\n".join([f"{e['provider']}: {e['error']}" for e in errors])
        raise APIClientError(f"All providers failed to get historical data for {symbol}:\n{error_details}")


# Create a global client instance
_client = FinancialDataClient()


# For backward compatibility with the existing API, with enhanced provider support
async def fetch_current_price(symbol: str, preferred_provider: Optional[str] = None) -> dict:
    """Fetches the latest price for a symbol using multiple providers with optional preferred provider.
    
    Args:
        symbol: Stock ticker symbol
        preferred_provider: Optional preferred data provider (falls back to others if not available)
        
    Returns:
        dict { symbol, price, timestamp, provider, additional_data }
    """
    return await _client.fetch_current_price(symbol, preferred_provider)


async def fetch_historical(symbol: str, start: datetime, end: datetime, 
                          preferred_provider: Optional[str] = None) -> dict:
    """Fetches daily OHLC series between `start` and `end` using multiple providers with optional preferred provider.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date for historical data
        end: End date for historical data
        preferred_provider: Optional preferred data provider (falls back to others if not available)
        
    Returns:
        dict { symbol, timeseries: [...], provider, start_date, end_date, metadata }
    """
    return await _client.fetch_historical(symbol, start, end, preferred_provider)
