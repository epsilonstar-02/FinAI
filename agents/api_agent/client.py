# Encapsulates HTTP calls to the upstream API, with error handling and retries.

from datetime import datetime

import httpx

from .config import settings


class APIClientError(Exception):
    pass


async def fetch_current_price(symbol: str) -> dict:
    """
    Fetches the latest price for a symbol.
    Returns dict { symbol, price, timestamp }.
    """
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": settings.API_KEY}
    async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
        response = await client.get(settings.BASE_URL, params=params)

    if response.status_code != 200:
        raise APIClientError(f"Upstream API returned {response.status_code}")

    data = response.json().get("Global Quote", {})
    if not data:
        raise APIClientError(f"No data for symbol {symbol}")

    price = float(data.get("05. price", 0))
    timestamp = datetime.utcnow()
    return {"symbol": symbol, "price": price, "timestamp": timestamp}


async def fetch_historical(symbol: str, start: datetime, end: datetime) -> dict:
    """
    Fetches daily OHLC series between `start` and `end`.
    Returns { symbol, timeseries: [...] }.
    """
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": settings.API_KEY,
    }
    async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
        response = await client.get(settings.BASE_URL, params=params)

    if response.status_code != 200:
        raise APIClientError(f"Upstream API returned {response.status_code}")

    raw = response.json().get("Time Series (Daily)", {})
    timeseries = []
    for date_str, daily in raw.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        if start.date() <= dt <= end.date():
            timeseries.append(
                {
                    "date": dt,
                    "open": float(daily["1. open"]),
                    "high": float(daily["2. high"]),
                    "low": float(daily["3. low"]),
                    "close": float(daily["4. close"]),
                }
            )
    timeseries.sort(key=lambda x: x["date"])
    return {"symbol": symbol, "timeseries": timeseries}
