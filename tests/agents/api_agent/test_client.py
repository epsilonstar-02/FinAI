import pytest
import respx
from httpx import Response
from agents.api_agent.client import fetch_current_price, fetch_historical, APIClientError
from agents.api_agent.config import settings
from datetime import datetime, date

# Base URL for mocking
BASE_URL = settings.BASE_URL
API_KEY = settings.API_KEY

@respx.mock
@pytest.mark.asyncio
async def test_fetch_current_price_success(monkeypatch):
    symbol = "AAPL"
    url = f"{BASE_URL}"
    # Mock the GLOBAL_QUOTE endpoint response
    respx.get(url).mock(
        return_value=Response(200, json={
            "Global Quote": {
                "05. price": "150.23"
            }
        })
    )
    result = await fetch_current_price(symbol)
    assert result["symbol"] == symbol
    assert isinstance(result["price"], float) and result["price"] == 150.23
    assert isinstance(result["timestamp"], datetime)

@respx.mock
@pytest.mark.asyncio
async def test_fetch_current_price_no_data(monkeypatch):
    symbol = "XXXX"
    url = f"{BASE_URL}"
    # Mock empty Global Quote
    respx.get(url).mock(return_value=Response(200, json={}))
    with pytest.raises(APIClientError):
        await fetch_current_price(symbol)

@respx.mock
@pytest.mark.asyncio
async def test_fetch_current_price_bad_status():
    symbol = "AAPL"
    url = f"{BASE_URL}"
    respx.get(url).mock(return_value=Response(500, json={}))
    with pytest.raises(APIClientError):
        await fetch_current_price(symbol)

@respx.mock
@pytest.mark.asyncio
async def test_fetch_historical_filters_and_sorts():
    symbol = "MSFT"
    # Prepare a fake full timeseries with multiple dates
    raw = {
        "2025-05-25": {"1. open": "100", "2. high": "110", "3. low": "90", "4. close": "105"},
        "2025-05-26": {"1. open": "106", "2. high": "112", "3. low": "101", "4. close": "110"},
        "2025-05-24": {"1. open": "98", "2. high": "102", "3. low": "95", "4. close": "100"}
    }
    url = f"{BASE_URL}"
    respx.get(url).mock(
        return_value=Response(200, json={"Time Series (Daily)": raw})
    )
    start = datetime(2025, 5, 25)
    end = datetime(2025, 5, 26)
    data = await fetch_historical(symbol, start, end)
    # Ensure returned times are sorted and within range
    dates = [pt["date"] for pt in data["timeseries"]]
    assert dates == [date(2025,5,25), date(2025,5,26)]
