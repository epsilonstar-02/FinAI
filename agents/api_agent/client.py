# agents/api_agent/client.py
# Significant refactoring for clarity, error handling, and async operations.

import httpx
from .config import settings, DataProvider
from datetime import datetime, timedelta, date # Added date
import yfinance as yf
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from alpha_vantage.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class ProviderNotAvailableError(APIClientError):
    """Raised when a specific data provider is not available or not configured."""
    pass


class NoDataAvailableError(APIClientError):
    """Raised when no data is available for the requested symbol from a provider."""
    pass


class FinancialDataProvider:
    """Base class for financial data providers."""
    
    def __init__(self, provider_name: DataProvider, timeout: int = settings.TIMEOUT):
        self.provider_name = provider_name.value # Store string value for logging/responses
        self.timeout = timeout
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol. Must return keys: symbol, price, timestamp, provider."""
        raise NotImplementedError
        
    async def get_historical_data(self, symbol: str, start: date, end: date) -> Dict[str, Any]:
        """Get historical data. Must return keys: symbol, timeseries, provider, start_date, end_date."""
        raise NotImplementedError


class YahooFinanceProvider(FinancialDataProvider):
    def __init__(self, timeout: int = settings.TIMEOUT):
        super().__init__(DataProvider.YAHOO_FINANCE, timeout)

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        try:
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            # Use 'fast_info' for potentially quicker current price or 'history' for last close
            # Fast info is less prone to timezone issues for "current" price if market is open
            info = await loop.run_in_executor(None, lambda: ticker.fast_info)
            
            price = info.last_price
            if price is None: # Fallback to previous close if last_price is not available
                history_data = await loop.run_in_executor(None, lambda: ticker.history(period="1d"))
                if history_data.empty or 'Close' not in history_data or history_data['Close'].empty:
                    raise NoDataAvailableError(f"No current price data for {symbol} from {self.provider_name}")
                price = history_data['Close'].iloc[-1]

            return {
                "symbol": symbol, 
                "price": float(price), 
                "timestamp": datetime.utcnow(), # Timestamp of fetch
                "provider": self.provider_name
            }
        except NoDataAvailableError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} error for {symbol} current price: {e}", exc_info=True)
            raise APIClientError(f"{self.provider_name} error processing {symbol}: {e}") from e
    
    async def get_historical_data(self, symbol: str, start: date, end: date) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        try:
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            # yfinance uses date strings for start/end
            data_df = await loop.run_in_executor(
                None, 
                lambda: ticker.history(start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d")) # Add 1 day to end to include it
            )
            
            if data_df.empty:
                raise NoDataAvailableError(f"No historical data for {symbol} ({start} to {end}) from {self.provider_name}")
                
            timeseries = []
            for date_idx, row in data_df.iterrows(): 
                # Ensure date_idx is a date object (it's usually a pandas Timestamp)
                dt = date_idx.date()
                if start <= dt <= end: # Filter to ensure within requested range
                    timeseries.append({
                        "date": dt, 
                        "open": float(row["Open"]), "high": float(row["High"]),
                        "low": float(row["Low"]), "close": float(row["Close"]),
                        "volume": int(row["Volume"])
                    })
            
            if not timeseries: # If filtering removed all data
                 raise NoDataAvailableError(f"No historical data for {symbol} in the exact range {start} to {end} after yfinance fetch from {self.provider_name}")

            # timeseries.sort(key=lambda x: x["date"]) # yfinance usually returns sorted data
            return {
                "symbol": symbol, "timeseries": timeseries, "provider": self.provider_name,
                "start_date": start, "end_date": end # Return requested range
            }
        except NoDataAvailableError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} historical data error for {symbol}: {e}", exc_info=True)
            raise APIClientError(f"{self.provider_name} error processing historical data for {symbol}: {e}") from e


class AlphaVantageProvider(FinancialDataProvider):
    def __init__(self, api_key: str, timeout: int = settings.TIMEOUT):
        super().__init__(DataProvider.ALPHA_VANTAGE, timeout)
        if not api_key:
            raise ProviderNotAvailableError(f"{self.provider_name} API key not configured.")
        self.api_key = api_key
        # The alpha_vantage library handles its own client/timeout implicitly
        self.client = TimeSeries(key=self.api_key, output_format='json', treat_info_as_error=True)
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        try:
            data, _ = await loop.run_in_executor(None, lambda: self.client.get_quote_endpoint(symbol=symbol))
            if not data or "05. price" not in data or data["05. price"] is None:
                raise NoDataAvailableError(f"No current price data for {symbol} from {self.provider_name}")
            price = float(data["05. price"])
            return {"symbol": symbol, "price": price, "timestamp": datetime.utcnow(), "provider": self.provider_name}
        except NoDataAvailableError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} error for {symbol} current price: {e}", exc_info=True)
            if "call frequency" in str(e).lower() or "premium endpoint" in str(e).lower():
                 raise APIClientError(f"{self.provider_name} API limit or access issue for {symbol}: {e}") from e
            raise APIClientError(f"{self.provider_name} error processing {symbol}: {e}") from e
    
    async def get_historical_data(self, symbol: str, start: date, end: date) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        try:
            # Alpha Vantage returns full history, then we filter
            data, _ = await loop.run_in_executor(
                None, lambda: self.client.get_daily_adjusted(symbol=symbol, outputsize='full')
            )
            if not data:
                raise NoDataAvailableError(f"No historical data for {symbol} from {self.provider_name}")
                
            timeseries = []
            for date_str, daily_data in data.items():
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if start <= dt <= end:
                        timeseries.append({
                            "date": dt, "open": float(daily_data["1. open"]),
                            "high": float(daily_data["2. high"]), "low": float(daily_data["3. low"]),
                            "close": float(daily_data["4. close"]), # "5. adjusted close" is also available
                            "volume": int(daily_data["6. volume"])
                        })
                except (ValueError, KeyError) as item_err:
                    logger.warning(f"Skipping malformed data point from {self.provider_name} for {symbol} on {date_str}: {item_err}")
            
            if not timeseries:
                 raise NoDataAvailableError(f"No data for {symbol} in range {start}-{end} from {self.provider_name} after filtering.")
            timeseries.sort(key=lambda x: x["date"])
            return {"symbol": symbol, "timeseries": timeseries, "provider": self.provider_name, "start_date": start, "end_date": end}
        except NoDataAvailableError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} historical data error for {symbol}: {e}", exc_info=True)
            if "call frequency" in str(e).lower() or "premium endpoint" in str(e).lower():
                 raise APIClientError(f"{self.provider_name} API limit or access issue for historical data for {symbol}: {e}") from e
            raise APIClientError(f"{self.provider_name} error processing historical data for {symbol}: {e}") from e


class FMPProvider(FinancialDataProvider):
    def __init__(self, api_key: str, base_url: str = settings.FMP_URL, timeout: int = settings.TIMEOUT):
        super().__init__(DataProvider.FMP, timeout)
        if not api_key:
            raise ProviderNotAvailableError(f"{self.provider_name} API key not configured.")
        self.api_key = api_key
        self.base_url = base_url
    
    async def _make_fmp_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apikey": self.api_key,**(params or {})}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=request_params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"{self.provider_name} HTTP error for {url}: {e.response.status_code} - {e.response.text}", exc_info=True)
                if e.response.status_code == 401 or e.response.status_code == 403: # Unauthorized or Forbidden
                    raise APIClientError(f"{self.provider_name} API key invalid or insufficient permissions for {url}")
                raise APIClientError(f"{self.provider_name} API request failed with status {e.response.status_code} for {url}") from e
            except httpx.RequestError as e: # Network errors
                logger.error(f"{self.provider_name} request error for {url}: {e}", exc_info=True)
                raise APIClientError(f"{self.provider_name} request failed for {url}: {e}") from e


    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        try:
            data = await self._make_fmp_request(f"quote/{symbol}")
            if not data or not isinstance(data, list) or not data[0].get("price"):
                raise NoDataAvailableError(f"No current price data for {symbol} from {self.provider_name}")
            price = float(data[0]["price"])
            return {"symbol": symbol, "price": price, "timestamp": datetime.utcnow(), "provider": self.provider_name}
        except NoDataAvailableError:
            raise
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"{self.provider_name} data parsing error for current price {symbol}: {e}", exc_info=True)
            raise APIClientError(f"{self.provider_name} error parsing current price data for {symbol}: {e}") from e
        except APIClientError: # Re-raise if it's already our specific error
            raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"{self.provider_name} unexpected error for current price {symbol}: {e}", exc_info=True)
            raise APIClientError(f"{self.provider_name} unexpected error processing {symbol}: {e}") from e

    async def get_historical_data(self, symbol: str, start: date, end: date) -> Dict[str, Any]:
        try:
            params = {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")}
            data = await self._make_fmp_request(f"historical-price-full/{symbol}", params=params)

            if not data or "historical" not in data or not data["historical"]:
                raise NoDataAvailableError(f"No historical data for {symbol} ({start} to {end}) from {self.provider_name}")
                
            timeseries = []
            for item in data["historical"]:
                try:
                    # FMP `date` field is already YYYY-MM-DD string
                    dt = datetime.strptime(item["date"], "%Y-%m-%d").date()
                    # Filter here as FMP might return slightly outside range depending on their API
                    if start <= dt <= end:
                        timeseries.append({
                            "date": dt, "open": float(item["open"]), "high": float(item["high"]),
                            "low": float(item["low"]), "close": float(item["close"]),
                            "volume": int(item["volume"])
                        })
                except (ValueError, TypeError, KeyError) as item_err:
                     logger.warning(f"Skipping malformed data point from FMP for {symbol} on {item.get('date', 'Unknown date')}: {item_err}")
            
            if not timeseries:
                 raise NoDataAvailableError(f"No data for {symbol} in range {start}-{end} from {self.provider_name} after filtering.")
            
            timeseries.sort(key=lambda x: x["date"])
            return {"symbol": symbol, "timeseries": timeseries, "provider": self.provider_name, "start_date": start, "end_date": end}
        except NoDataAvailableError:
            raise
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"{self.provider_name} data parsing error for historical data {symbol}: {e}", exc_info=True)
            raise APIClientError(f"{self.provider_name} error parsing historical data for {symbol}: {e}") from e
        except APIClientError:
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} unexpected error for historical data {symbol}: {e}", exc_info=True)
            raise APIClientError(f"{self.provider_name} unexpected error processing historical data for {symbol}: {e}") from e


def get_current_max_retries() -> int:
    return settings.MAX_RETRIES
    
def get_current_retry_backoff() -> float:
    return settings.RETRY_BACKOFF

class FinancialDataClient:
    def __init__(self):
        self.providers: Dict[str, FinancialDataProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        self.providers.clear()
        logger.info(f"Initializing financial data providers. Timeout: {settings.TIMEOUT}s")

        # Yahoo Finance (always available)
        self.providers[DataProvider.YAHOO_FINANCE.value] = YahooFinanceProvider(timeout=settings.TIMEOUT)
        
        # Alpha Vantage (if key is set)
        if settings.ALPHA_VANTAGE_KEY:
            try:
                self.providers[DataProvider.ALPHA_VANTAGE.value] = AlphaVantageProvider(
                    api_key=settings.ALPHA_VANTAGE_KEY, timeout=settings.TIMEOUT
                )
            except ProviderNotAvailableError as e:
                 logger.warning(f"Could not initialize AlphaVantageProvider: {e}")
        else:
            logger.info("Alpha Vantage API key not set. Provider will be unavailable.")

        # FMP (if key is set)
        if settings.FMP_KEY:
            try:
                self.providers[DataProvider.FMP.value] = FMPProvider(
                    api_key=settings.FMP_KEY, base_url=settings.FMP_URL, timeout=settings.TIMEOUT
                )
            except ProviderNotAvailableError as e:
                logger.warning(f"Could not initialize FMPProvider: {e}")
        else:
            logger.info("FMP API key not set. Provider will be unavailable.")
            
        logger.info(f"Initialized providers: {list(self.providers.keys())}")

    async def _try_provider_method(self, provider_name_str: str, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        if provider_name_str not in self.providers:
            raise ProviderNotAvailableError(f"Provider '{provider_name_str}' not available or not initialized.")
        
        provider_instance = self.providers[provider_name_str]
        method_to_call = getattr(provider_instance, method_name)
        # logger.debug(f"Calling {method_name} on {provider_name_str} with args: {args}, kwargs: {kwargs}")
        return await method_to_call(*args, **kwargs)

    async def _fetch_data_with_priority_fallback(
        self, 
        method_name: str, 
        symbol: str, 
        preferred_provider_str: Optional[str] = None,
        additional_args: Optional[tuple] = None
    ) -> Dict[str, Any]:
        errors_log = []
        
        # Build provider list: preferred, then priority list, then any other available
        providers_to_try_ordered: List[str] = []
        
        if preferred_provider_str and preferred_provider_str in self.providers:
            providers_to_try_ordered.append(preferred_provider_str)
        elif preferred_provider_str:
            logger.warning(f"Preferred provider '{preferred_provider_str}' not available. Proceeding with default priority.")

        for p_enum_member in settings.PROVIDER_PRIORITY:
            p_str = p_enum_member.value
            if p_str in self.providers and p_str not in providers_to_try_ordered:
                providers_to_try_ordered.append(p_str)
        
        # Add any other available providers not yet in the list
        for p_str in self.providers.keys():
            if p_str not in providers_to_try_ordered:
                providers_to_try_ordered.append(p_str)

        if not providers_to_try_ordered:
            raise APIClientError(f"No data providers available to fetch {method_name} for {symbol}.")

        logger.debug(f"Attempting {method_name} for '{symbol}' using providers in order: {providers_to_try_ordered}")
        
        call_args = (symbol,) + (additional_args or tuple())

        for provider_name_str in providers_to_try_ordered:
            try:
                logger.debug(f"Trying provider '{provider_name_str}' for {method_name} on '{symbol}'...")
                result = await self._try_provider_method(provider_name_str, method_name, *call_args)
                logger.info(f"Successfully fetched {method_name} for '{symbol}' from '{provider_name_str}'.")
                return result # Success
            except NoDataAvailableError as e:
                logger.info(f"Provider '{provider_name_str}' has no {method_name} data for '{symbol}': {e}")
                errors_log.append({"provider": provider_name_str, "error_type": "NoDataAvailableError", "message": str(e)})
            except APIClientError as e: # Includes ProviderNotAvailableError, other client errors from provider
                logger.warning(f"Provider '{provider_name_str}' failed for {method_name} on '{symbol}': {type(e).__name__} - {e}")
                errors_log.append({"provider": provider_name_str, "error_type": type(e).__name__, "message": str(e)})
            except Exception as e:
                logger.error(
                    f"Unexpected error from provider '{provider_name_str}' for {method_name} on '{symbol}': {type(e).__name__} - {e}", 
                    exc_info=True
                )
                errors_log.append({"provider": provider_name_str, "error_type": "UnexpectedError", "message": str(e)})

            if not settings.ENABLE_FALLBACK and len(providers_to_try_ordered) > 1 and providers_to_try_ordered[0] == provider_name_str :
                logger.info(f"Fallback disabled and preferred/first provider '{provider_name_str}' failed. Stopping.")
                break 
        
        error_details = "\n".join([f"- {e['provider']} ({e['error_type']}): {e['message']}" for e in errors_log])
        final_error_message = f"All attempted providers failed to get {method_name} for '{symbol}'. Errors:\n{error_details}"
        
        # If the last error was NoDataAvailableError and fallback was disabled or only one provider tried
        if errors_log and errors_log[-1]["error_type"] == "NoDataAvailableError":
            if not settings.ENABLE_FALLBACK or len(providers_to_try_ordered) == 1:
                raise NoDataAvailableError(final_error_message) # Make it clear no data from the only attempted provider
        raise APIClientError(final_error_message) # General failure after trying all options
    
    @retry(
        retry=retry_if_exception_type(APIClientError), # Retry on general APIClientError
        stop=stop_after_attempt(get_current_max_retries),
        wait=wait_exponential(multiplier=get_current_retry_backoff, min=1, max=10), # Min/max for wait
        reraise=True # Reraise the exception after retries are exhausted
    )
    async def fetch_current_price(self, symbol: str, preferred_provider_str: Optional[str] = None) -> Dict[str, Any]:
        return await self._fetch_data_with_priority_fallback("get_current_price", symbol, preferred_provider_str)
    
    @retry(
        retry=retry_if_exception_type(APIClientError),
        stop=stop_after_attempt(get_current_max_retries),
        wait=wait_exponential(multiplier=get_current_retry_backoff, min=1, max=10),
        reraise=True
    )
    async def fetch_historical_data(self, symbol: str, start: date, end: date, 
                               preferred_provider_str: Optional[str] = None) -> Dict[str, Any]:
        return await self._fetch_data_with_priority_fallback(
            "get_historical_data", symbol, preferred_provider_str, additional_args=(start, end)
        )

    async def fetch_from_specific_provider(self, provider_name_str: str, method_name: str, *args) -> Dict[str, Any]:
        """Fetches directly from a specific provider, bypassing fallback and client-level retry logic."""
        try:
            return await self._try_provider_method(provider_name_str, method_name, *args)
        except (NoDataAvailableError, ProviderNotAvailableError, APIClientError):
            raise 
        except Exception as e:
            logger.error(
                f"Unexpected error calling {method_name} on provider '{provider_name_str}' with args {args}: {e}",
                exc_info=True
            )
            raise APIClientError(f"Unexpected error from provider '{provider_name_str}': {e}") from e


_financial_data_client_instance: Optional[FinancialDataClient] = None

def get_financial_data_client() -> FinancialDataClient:
    """Returns a singleton instance of the FinancialDataClient."""
    global _financial_data_client_instance
    if _financial_data_client_instance is None:
        _financial_data_client_instance = FinancialDataClient()
    return _financial_data_client_instance

async def fetch_current_price_with_fallback(symbol: str, preferred_provider_str: Optional[str] = None) -> Dict[str, Any]:
    client = get_financial_data_client()
    return await client.fetch_current_price(symbol, preferred_provider_str)

async def fetch_historical_data_with_fallback(symbol: str, start: date, end: date, 
                                              preferred_provider_str: Optional[str] = None) -> Dict[str, Any]:
    client = get_financial_data_client()
    return await client.fetch_historical_data(symbol, start, end, preferred_provider_str)

async def fetch_current_price_from_specific_provider(symbol: str, provider_str: str) -> Dict[str, Any]:
    client = get_financial_data_client()
    return await client.fetch_from_specific_provider(provider_str, "get_current_price", symbol)

# For re-initialization, e.g. in tests or if settings change
def reinitialize_financial_data_client():
    global _financial_data_client_instance
    logger.info("Re-initializing financial data client...")
    _financial_data_client_instance = FinancialDataClient()
    return _financial_data_client_instance