from datetime import datetime

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from agents.api_agent.client import (APIClientError, fetch_current_price,
                                     fetch_historical)
from agents.api_agent.main import app


# --- Fixtures to monkeypatch client calls ---
@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    # Ensure no real API key is needed
    monkeypatch.setenv("ALPHAVANTAGE_KEY", "TESTKEY")
    yield


@pytest.fixture
def price_success(monkeypatch):
    async def fake_price(symbol):
        return {"symbol": symbol, "price": 123.45, "timestamp": datetime.utcnow()}

    # Patch the function where it's imported in the main module
    monkeypatch.setattr("agents.api_agent.main.fetch_current_price", fake_price)
    return fake_price


@pytest.fixture
def historical_success(monkeypatch):
    async def fake_hist(symbol, start, end):
        return {"symbol": symbol, "timeseries": []}

    # Patch the function where it's imported in the main module
    monkeypatch.setattr("agents.api_agent.main.fetch_historical", fake_hist)
    return fake_hist


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"status": "ok", "agent": "API Agent"}


def test_price_endpoint_success(client, price_success):
    r = client.get("/price", params={"symbol": "GOOG"})
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["symbol"] == "GOOG"
    assert body["price"] == 123.45


def test_price_endpoint_error(client, monkeypatch):
    async def raise_error(symbol):
        raise APIClientError("Upstream error")

    monkeypatch.setattr("agents.api_agent.client.fetch_current_price", raise_error)
    r = client.get("/price", params={"symbol": "BAD"})
    assert r.status_code == status.HTTP_502_BAD_GATEWAY


def test_historical_endpoint_date_validation(client, historical_success):
    r = client.get(
        "/historical",
        params={"symbol": "TSLA", "start": "2025-05-27", "end": "2025-05-26"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST


def test_historical_endpoint_success(client, historical_success):
    r = client.get(
        "/historical",
        params={"symbol": "TSLA", "start": "2025-05-25", "end": "2025-05-26"},
    )
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"symbol": "TSLA", "timeseries": []}
