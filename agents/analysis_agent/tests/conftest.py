import pytest
from fastapi.testclient import TestClient
from typing import List, Dict, Generator
import time

# Adjust the import path based on how you run pytest
# If running pytest from the root directory of the project (containing 'agents'):
from agents.analysis_agent.main import app, settings as app_settings
from agents.analysis_agent.models import HistoricalDataPoint

# If the above imports don't work, you might need to adjust PYTHONPATH
# or use relative imports if running pytest from within the 'analysis_agent' directory.

@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Fixture to create a TestClient for the FastAPI application.
    """
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_prices() -> Dict[str, float]:
    """
    Fixture for sample current prices.
    """
    return {"AAPL": 150.0, "MSFT": 300.0}

@pytest.fixture
def sample_historical_data_point(date_str: str = "2023-01-01", close_price: float = 100.0) -> HistoricalDataPoint:
    """
    Fixture for a single historical data point.
    Allows customization via indirect parametrization if needed.
    """
    return HistoricalDataPoint(date=date_str, close=close_price)

@pytest.fixture
def sample_historical_data_list_short() -> List[HistoricalDataPoint]:
    """
    Fixture for a short list of historical data points (2 points).
    """
    return [
        HistoricalDataPoint(date="2023-01-02", close=102.0), # Newest
        HistoricalDataPoint(date="2023-01-01", close=100.0)  # Oldest
    ]

@pytest.fixture
def sample_historical_data_list_long() -> List[HistoricalDataPoint]:
    """
    Fixture for a longer list of historical data points for volatility/correlation.
    """
    return [
        HistoricalDataPoint(date="2023-01-05", close=105.0),
        HistoricalDataPoint(date="2023-01-04", close=103.0),
        HistoricalDataPoint(date="2023-01-03", close=104.0),
        HistoricalDataPoint(date="2023-01-02", close=102.0),
        HistoricalDataPoint(date="2023-01-01", close=100.0),
    ]

@pytest.fixture
def sample_historical_data_dict(
    sample_historical_data_list_short: List[HistoricalDataPoint],
    sample_historical_data_list_long: List[HistoricalDataPoint]
) -> Dict[str, List[HistoricalDataPoint]]:
    """
    Fixture for a dictionary of historical data.
    """
    # Create distinct data for MSFT to avoid identical correlations/volatilities
    msft_data = [
        HistoricalDataPoint(date="2023-01-05", close=310.0),
        HistoricalDataPoint(date="2023-01-04", close=305.0),
        HistoricalDataPoint(date="2023-01-03", close=308.0),
        HistoricalDataPoint(date="2023-01-02", close=302.0),
        HistoricalDataPoint(date="2023-01-01", close=300.0),
    ]
    return {
        "AAPL": sample_historical_data_list_long,
        "MSFT": msft_data
    }

@pytest.fixture
def sample_analyze_request_payload(
    sample_prices: Dict[str, float],
    sample_historical_data_dict: Dict[str, List[HistoricalDataPoint]]
) -> Dict:
    """
    Fixture for a sample AnalyzeRequest payload.
    """
    # Convert HistoricalDataPoint objects to dicts for the JSON payload
    historical_payload = {
        symbol: [hdp.model_dump() for hdp in hdp_list]
        for symbol, hdp_list in sample_historical_data_dict.items()
    }
    return {
        "prices": sample_prices,
        "historical": historical_payload,
        "provider": "default",
        "include_correlations": False,
        "include_risk_metrics": False
    }

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Automatically mock settings for consistent test behavior.
    Disables cache and rate limiting by default for most tests.
    Individual tests can re-enable them if needed.
    """
    monkeypatch.setattr(app_settings, 'CACHE_ENABLED', False)
    monkeypatch.setattr(app_settings, 'RATE_LIMIT_ENABLED', False)
    monkeypatch.setattr(app_settings, 'VOLATILITY_WINDOW', 5) # Smaller window for easier testing
    monkeypatch.setattr(app_settings, 'ALERT_THRESHOLD', 0.05)

@pytest.fixture
def mock_time(monkeypatch):
    """Fixture to mock time.time()"""
    mocked_time = time.time()

    class MockTime:
        def time(self):
            return mocked_time
        
        def sleep(self, seconds):
            nonlocal mocked_time
            mocked_time += seconds

        def advance_time(self, seconds):
            nonlocal mocked_time
            mocked_time += seconds
            
    mt = MockTime()
    monkeypatch.setattr(time, 'time', mt.time)
    monkeypatch.setattr(time, 'sleep', mt.sleep) # If any code uses time.sleep
    return mt

@pytest.fixture
def long_historical_data_fixture() -> Dict[str, List[HistoricalDataPoint]]:
    """
    Fixture that generates a longer series of historical data points for testing volatility and risk metrics.
    Returns a dictionary with symbols as keys and lists of HistoricalDataPoint as values.
    """
    import numpy as np
    import pandas as pd
    
    data = {}
    base_date = pd.to_datetime("2023-01-01")  # pandas for robust date handling
    
    # Helper function to create a single data point
    HDP = HistoricalDataPoint
    
    for symbol in ["SYM1", "SYM2"]:
        history = []
        price = 100.0
        for i in range(60):  # Generate 60 data points
            date_obj = base_date + pd.Timedelta(days=i)
            date_str = date_obj.strftime('%Y-%m-%d')
            
            # Add some noise, ensure price doesn't go to zero or negative easily
            price_change = np.random.uniform(-1, 1)
            # Ensure price stays positive for simplicity in pct_change calculations
            new_price = max(0.1, price + price_change) 
            history.append(HDP(date=date_str, close=new_price))
            price = new_price
            
        # Oldest first; providers internally sort by date reverse=True
        data[symbol] = history
        
    return data