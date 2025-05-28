"""Tests for the Analysis Agent main module."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import httpx

from agents.analysis_agent.main import app
from agents.analysis_agent.models import AnalyzeRequest, AnalyzeResponse, HistoricalDataPoint

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health endpoint."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["agent"] == "Analysis Agent"

@pytest.mark.asyncio
@patch("agents.analysis_agent.main.build_summary") # Applied fourth, so first arg
@patch("agents.analysis_agent.main.compute_volatility") # Applied third, so second arg
@patch("agents.analysis_agent.main.compute_changes") # Applied second, so third arg
@patch("agents.analysis_agent.main.compute_exposures") # Applied first, so fourth arg
async def test_analyze_endpoint(
    mock_compute_exposures, # Corresponds to patch for compute_exposures
    mock_compute_changes,   # Corresponds to patch for compute_changes
    mock_compute_volatility,# Corresponds to patch for compute_volatility
    mock_build_summary,     # Corresponds to patch for build_summary
):
    """Test the analyze endpoint with mocked calculator functions."""
    # Setup mock returns
    mock_compute_exposures.return_value = {"AAPL": 0.5, "MSFT": 0.5}
    mock_compute_changes.return_value = {"AAPL": 0.05, "MSFT": -0.03}
    mock_compute_volatility.return_value = {"AAPL": 0.02, "MSFT": 0.04}
    mock_build_summary.return_value = "Analysis Summary:\n- Significant price changes: AAPL"
    
    # Test request data
    request_data = {
        "prices": {"AAPL": 150.0, "MSFT": 300.0},
        "historical": {
            "AAPL": [
                {"date": "2023-01-02", "close": 150.0}, # Passed as dicts, Pydantic will convert
                {"date": "2023-01-01", "close": 140.0}
            ],
            "MSFT": [
                {"date": "2023-01-02", "close": 300.0},
                {"date": "2023-01-01", "close": 310.0}
            ]
        }
    }
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "exposures" in data
        assert "changes" in data
        assert "volatility" in data
        assert "summary" in data
        
        # Verify mock calls (Pydantic converts dicts to HistoricalDataPoint objects)
        # So the arguments to compute_X functions will have these objects
        mock_compute_exposures.assert_called_once_with(request_data["prices"])
        
        # For calls with HistoricalDataPoint, direct comparison of args is tricky.
        # We can check if it was called, and a more detailed check would involve
        # asserting properties of the arguments if necessary.
        mock_compute_changes.assert_called_once()
        # Example of checking args if needed:
        # args_changes, _ = mock_compute_changes.call_args
        # assert isinstance(args_changes[0]["AAPL"][0], HistoricalDataPoint)

        mock_compute_volatility.assert_called_once()
        # args_volatility, _ = mock_compute_volatility.call_args
        # assert isinstance(args_volatility[0]["AAPL"][0], HistoricalDataPoint)
        # assert args_volatility[1] == settings.VOLATILITY_WINDOW # from config

        mock_build_summary.assert_called_once()


@pytest.mark.asyncio
@patch("agents.analysis_agent.main.compute_changes")
@patch("agents.analysis_agent.main.compute_exposures")
async def test_analyze_endpoint_error(
    mock_compute_exposures_actual,
    mock_compute_changes_actual
):
    """Test error handling in the analyze endpoint by ensuring the global
    exception handler returns the correct JSON 500 response."""

    # Setup mocks: compute_exposures (called first in the endpoint) will raise an error.
    mock_compute_exposures_actual.side_effect = Exception("Test calculation error")

    request_data = {
        "prices": {"AAPL": 150.0},
        "historical": {
            "AAPL": [
                HistoricalDataPoint(date="2023-01-01", close=140.0).model_dump(),
                HistoricalDataPoint(date="2023-01-02", close=145.0).model_dump()
            ]
        }
    }

    # Explicitly create an ASGITransport with raise_app_exceptions=False
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/analyze", json=request_data)

        assert response.status_code == 500
        data = response.json()

        assert data["status"] == "error"
        assert data["message"] == "An unexpected error occurred"
        assert "Test calculation error" in data["details"]

    mock_compute_exposures_actual.assert_called_once()
    mock_compute_changes_actual.assert_not_called()