"""Tests for the Analysis Agent calculator module."""
import pytest
import numpy as np
from agents.analysis_agent.calculator import (
    compute_exposures,
    compute_changes,
    compute_volatility,
    build_summary,
)
from agents.analysis_agent.models import HistoricalDataPoint # Import the model

def test_compute_exposures():
    """Test compute_exposures function with sample data."""
    # Test case 1: Normal case
    prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 300.0}
    expected = {"AAPL": 100.0/600.0, "MSFT": 200.0/600.0, "GOOG": 300.0/600.0}
    result = compute_exposures(prices)
    assert result == pytest.approx(expected)
    
    # Test case 2: Empty prices
    assert compute_exposures({}) == {}
    
    # Test case 3: Zero prices
    prices_zero = {"AAPL": 0.0, "MSFT": 0.0}
    expected_zero = {"AAPL": 0.0, "MSFT": 0.0}
    assert compute_exposures(prices_zero) == expected_zero

def test_compute_changes():
    """Test compute_changes function with sample data."""
    # Test case 1: Normal case
    historical = {
        "AAPL": [
            HistoricalDataPoint(date="2023-01-02", close=110.0),
            HistoricalDataPoint(date="2023-01-01", close=100.0),
        ],
        "MSFT": [
            HistoricalDataPoint(date="2023-01-02", close=180.0),
            HistoricalDataPoint(date="2023-01-01", close=200.0),
        ],
    }
    expected = {"AAPL": 0.1, "MSFT": -0.1}
    result = compute_changes(historical)
    assert result["AAPL"] == pytest.approx(expected["AAPL"])
    assert result["MSFT"] == pytest.approx(expected["MSFT"])
    
    # Test case 2: Empty historical data
    assert compute_changes({}) == {}
    
    # Test case 3: Single data point (no change calculable)
    single_point = {"AAPL": [HistoricalDataPoint(date="2023-01-01", close=100.0)]}
    assert compute_changes(single_point) == {"AAPL": 0.0}
    
    # Test case 4: Zero previous price
    zero_previous = {
        "AAPL": [
            HistoricalDataPoint(date="2023-01-02", close=100.0),
            HistoricalDataPoint(date="2023-01-01", close=0.0),
        ],
    }
    assert compute_changes(zero_previous) == {"AAPL": 0.0}

    # Test case 5: Zero current price, non-zero previous
    zero_current = {
        "XYZ": [
            HistoricalDataPoint(date="2023-01-02", close=0.0),
            HistoricalDataPoint(date="2023-01-01", close=50.0),
        ]
    }
    assert compute_changes(zero_current) == {"XYZ": -1.0}


def test_compute_volatility():
    """Test compute_volatility function with sample data."""
    # Test case 1: Normal case
    historical_data_aapl = [
        HistoricalDataPoint(date="2023-01-05", close=105.0), # Newest
        HistoricalDataPoint(date="2023-01-04", close=100.0),
        HistoricalDataPoint(date="2023-01-03", close=95.0),
        HistoricalDataPoint(date="2023-01-02", close=90.0),
        HistoricalDataPoint(date="2023-01-01", close=85.0),  # Oldest
    ]
    historical = {"AAPL": historical_data_aapl}
    
    # Expected returns: (105-100)/100=0.05, (100-95)/95=0.05263, (95-90)/90=0.05556, (90-85)/85=0.05882
    # These are returns relative to the previous day's close.
    # P_newest, P_next_newest, ..., P_oldest
    # returns = [ (P0-P1)/P1, (P1-P2)/P2, ... ]
    # For AAPL data: prices are [105, 100, 95, 90, 85] (newest to oldest)
    # returns are:
    # r1 = (105 - 100) / 100 = 0.05
    # r2 = (100 - 95) / 95   = 0.0526315789
    # r3 = (95 - 90) / 90    = 0.0555555555
    # r4 = (90 - 85) / 85    = 0.0588235294
    expected_returns = [0.05, 5/95, 5/90, 5/85]
    expected_volatility_aapl = np.std(expected_returns)
    
    result = compute_volatility(historical, window=5) # Window of 5 prices gives 4 returns
    assert result["AAPL"] == pytest.approx(expected_volatility_aapl)
    
    # Test case 2: Empty historical data
    assert compute_volatility({}, window=5) == {}
    
    # Test case 3: Window larger than available data, but still enough for 1 return
    small_data = {
        "MSFT": [
            HistoricalDataPoint(date="2023-01-02", close=110.0),
            HistoricalDataPoint(date="2023-01-01", close=100.0),
        ],
    } # 2 data points -> 1 return
    # returns = [(110-100)/100] = [0.1]
    # std([0.1]) = 0.0 (as std dev of a single number is 0)
    result_small = compute_volatility(small_data, window=10)
    assert result_small["MSFT"] == pytest.approx(0.0) 
    
    # Test case 4: Single data point (not enough for any returns)
    single_point = {"GOOG": [HistoricalDataPoint(date="2023-01-01", close=100.0)]}
    assert compute_volatility(single_point, window=5) == {"GOOG": 0.0}

    # Test case 5: Not enough data for any returns within window
    not_enough_for_return = {"XYZ": [HistoricalDataPoint(date="2023-01-01", close=100.0)]}
    assert compute_volatility(not_enough_for_return, window=2) == {"XYZ": 0.0}

    # Test case 6: Prices causing zero division in returns calculation (previous price is 0)
    zero_prev_price_hist = {
        "ZERO": [
            HistoricalDataPoint(date="2023-01-03", close=10.0),
            HistoricalDataPoint(date="2023-01-02", close=5.0),
            HistoricalDataPoint(date="2023-01-01", close=0.0), # This will cause previous price to be 0 for one return
        ]
    }
    # Prices: [10, 5, 0]
    # Returns: (10-5)/5 = 1.0; (5-0)/0 -> 0.0 (handled by code)
    # std([1.0, 0.0])
    expected_vol_zero = np.std([1.0, 0.0])
    result_zero = compute_volatility(zero_prev_price_hist, window=3)
    assert result_zero["ZERO"] == pytest.approx(expected_vol_zero)


def test_build_summary():
    """Test build_summary function with sample data."""
    exposures = {"AAPL": 0.2, "MSFT": 0.3, "GOOG": 0.5} # GOOG, MSFT, AAPL > 0.05
    changes = {"AAPL": 0.06, "MSFT": 0.02, "GOOG": -0.01} # AAPL abs(0.06) > 0.05
    volatility = {"AAPL": 0.03, "MSFT": 0.07, "GOOG": 0.04} # MSFT 0.07 > 0.05
    threshold = 0.05
    
    # Test case 1: Normal case with threshold 0.05
    # Expected sorted: AAPL, GOOG, MSFT for exposure
    # Expected sorted: AAPL for changes
    # Expected sorted: MSFT for volatility
    result = build_summary(exposures, changes, volatility, threshold=threshold)
    
    expected_summary_lines = [
        "Analysis Summary:",
        "- High exposure assets: AAPL, GOOG, MSFT", # Sorted
        "- Significant price changes: AAPL",      # Sorted (single item)
        "- High volatility assets: MSFT"          # Sorted (single item)
    ]
    expected_summary = "\n".join(expected_summary_lines)
    assert result == expected_summary
    
    # Test case 2: No alerts (high threshold)
    result_no_alerts = build_summary(exposures, changes, volatility, threshold=1.0) # High threshold
    expected_no_alerts_summary = "Analysis Summary:\n- No significant alerts detected."
    assert result_no_alerts == expected_no_alerts_summary
    
    # Test case 3: All alerts (low threshold)
    # Exposures: All (AAPL, GOOG, MSFT)
    # Changes: AAPL (0.06), MSFT (0.02) > 0.01
    # Volatility: AAPL (0.03), GOOG (0.04), MSFT (0.07) > 0.01
    result_all_alerts = build_summary(exposures, changes, volatility, threshold=0.01)
    expected_all_alerts_lines = [
        "Analysis Summary:",
        "- High exposure assets: AAPL, GOOG, MSFT",
        "- Significant price changes: AAPL, MSFT", # MSFT |0.02| > 0.01
        "- High volatility assets: AAPL, GOOG, MSFT"  # All > 0.01
    ]
    expected_all_summary = "\n".join(expected_all_alerts_lines)
    assert result_all_alerts == expected_all_summary