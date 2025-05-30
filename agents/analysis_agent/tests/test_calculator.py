import pytest
import numpy as np
from typing import Dict, List

# Adjust import paths as necessary
from agents.analysis_agent.calculator import (
    compute_exposures,
    compute_changes,
    compute_volatility,
    build_summary
)
from agents.analysis_agent.models import HistoricalDataPoint

# Helper to create HistoricalDataPoint objects easily
def HDP(date: str, close: float) -> HistoricalDataPoint:
    return HistoricalDataPoint(date=date, close=close)

class TestComputeExposures:
    def test_empty_prices(self):
        assert compute_exposures({}) == {}

    def test_single_asset(self):
        assert compute_exposures({"AAPL": 100.0}) == {"AAPL": 1.0}

    def test_multiple_assets(self):
        prices = {"AAPL": 100.0, "MSFT": 300.0}
        expected = {"AAPL": 0.25, "MSFT": 0.75} # 100/400, 300/400
        assert compute_exposures(prices) == expected

    def test_prices_sum_to_zero(self):
        prices = {"AAPL": 0.0, "MSFT": 0.0}
        expected = {"AAPL": 0.0, "MSFT": 0.0}
        assert compute_exposures(prices) == expected

    def test_one_price_zero(self):
        prices = {"AAPL": 100.0, "MSFT": 0.0}
        expected = {"AAPL": 1.0, "MSFT": 0.0}
        assert compute_exposures(prices) == expected

class TestComputeChanges:
    def test_empty_historical(self):
        assert compute_changes({}) == {}

    def test_single_asset_empty_history(self):
        assert compute_changes({"AAPL": []}) == {"AAPL": 0.0}

    def test_single_asset_one_point(self):
        history = {"AAPL": [HDP("2023-01-01", 100.0)]}
        assert compute_changes(history) == {"AAPL": 0.0}

    def test_single_asset_two_points_positive_change(self):
        # Newest first in list for easier reading, but function sorts
        history = {"AAPL": [HDP("2023-01-01", 100.0), HDP("2023-01-02", 110.0)]}
        # Sorted: (02, 110), (01, 100). Change = (110-100)/100 = 0.1
        assert compute_changes(history) == {"AAPL": 0.1}

    def test_single_asset_two_points_negative_change(self):
        history = {"AAPL": [HDP("2023-01-02", 90.0), HDP("2023-01-01", 100.0)]}
        # Sorted: (02, 90), (01, 100). Change = (90-100)/100 = -0.1
        # Correction: Input history should be sorted by function.
        # Sorted based on date: (02, 90), (01, 100) -> (90-100)/100 = -0.1. This is wrong.
        # sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
        # history[0] is newest, history[1] is previous.
        # Date "2023-01-02" > "2023-01-01"
        # So, current_close = 90 (from 2023-01-02), previous_close = 100 (from 2023-01-01)
        # (90 - 100) / 100 = -0.1
        assert pytest.approx(compute_changes(history)["AAPL"]) == -0.1

    def test_single_asset_two_points_zero_change(self):
        history = {"AAPL": [HDP("2023-01-02", 100.0), HDP("2023-01-01", 100.0)]}
        assert compute_changes(history) == {"AAPL": 0.0}
        
    def test_previous_close_zero(self):
        history = {"AAPL": [HDP("2023-01-02", 10.0), HDP("2023-01-01", 0.0)]}
        assert compute_changes(history) == {"AAPL": 0.0}

    def test_unsorted_historical_data(self):
        history = {"AAPL": [HDP("2023-01-01", 100.0), HDP("2023-01-03", 120.0), HDP("2023-01-02", 110.0)]}
        # Sorted: (03,120), (02,110), (01,100)
        # Current = 120, Previous = 110. Change = (120-110)/110
        expected_change = (120.0 - 110.0) / 110.0
        assert pytest.approx(compute_changes(history)["AAPL"]) == expected_change

class TestComputeVolatility:
    WINDOW_SIZE = 3 # Use a small window for testing

    def test_empty_historical(self):
        assert compute_volatility({}, self.WINDOW_SIZE) == {}

    def test_single_asset_empty_history(self):
        assert compute_volatility({"AAPL": []}, self.WINDOW_SIZE) == {"AAPL": 0.0}

    def test_single_asset_less_than_2_points(self):
        history = {"AAPL": [HDP("2023-01-01", 100.0)]}
        assert compute_volatility(history, self.WINDOW_SIZE) == {"AAPL": 0.0}

    def test_single_asset_less_than_2_points_in_window(self):
        # Window is 3, but only 1 point available.
        history = {"AAPL": [HDP("2023-01-01", 100.0)]}
        # sorted_history[:min(window, len(sorted_history))]
        # window_data will have 1 point. len(window_data) < 2 will be true.
        assert compute_volatility(history, self.WINDOW_SIZE) == {"AAPL": 0.0}

        # 2 points available, window_data will have 2 points. len(prices)=2, returns has 1 element.
        history_2pts = {"AAPL": [HDP("2023-01-02", 100.0), HDP("2023-01-01", 100.0)]}
        # prices = [100, 100], returns = [(100-100)/100] = [0.0]
        # np.std([0.0]) = 0.0
        assert compute_volatility(history_2pts, self.WINDOW_SIZE)["AAPL"] == 0.0
        
    def test_sufficient_data_points_constant_price(self):
        history = {"AAPL": [
            HDP("2023-01-03", 100.0), HDP("2023-01-02", 100.0), HDP("2023-01-01", 100.0)
        ]}
        # Prices (newest to oldest): [100, 100, 100]
        # Returns: [(100-100)/100, (100-100)/100] = [0.0, 0.0]
        # np.std([0.0, 0.0]) = 0.0
        assert compute_volatility(history, self.WINDOW_SIZE)["AAPL"] == 0.0

    def test_sufficient_data_points_varying_price(self):
        history = {"AAPL": [
            HDP("2023-01-04", 103.0), HDP("2023-01-03", 101.0), 
            HDP("2023-01-02", 102.0), HDP("2023-01-01", 100.0)
        ]} # 4 points
        # Window size is 3. sorted_history uses all 4. window_data takes newest 3 points.
        # Dates: 04, 03, 02. Prices (newest to oldest): [103, 101, 102]
        # Returns:
        # r1 = (103-101)/101 = 2/101 approx 0.0198
        # r2 = (101-102)/102 = -1/102 approx -0.0098
        returns = [(103.0-101.0)/101.0, (101.0-102.0)/102.0]
        expected_vol = np.std(returns)
        assert pytest.approx(compute_volatility(history, self.WINDOW_SIZE)["AAPL"]) == expected_vol

    def test_window_larger_than_data(self):
        history = {"AAPL": [HDP("2023-01-02", 110.0), HDP("2023-01-01", 100.0)]} # 2 points
        # Window size 3. window_data will take min(3, 2) = 2 points.
        # Prices: [110, 100]. Returns: [(110-100)/100] = [0.1]
        # np.std([0.1]) = 0.0
        assert compute_volatility(history, self.WINDOW_SIZE)["AAPL"] == 0.0

    def test_previous_close_zero_in_returns(self):
        history = {"AAPL": [
            HDP("2023-01-03", 10.0), HDP("2023-01-02", 0.0), HDP("2023-01-01", 5.0)
        ]}
        # Prices (newest to oldest): [10, 0, 5]
        # Returns:
        # r1 = (10-0)/0 -> 0.0 (due to if prices[i+1] != 0 else 0.0)
        # r2 = (0-5)/5 = -1.0
        returns = [0.0, -1.0]
        expected_vol = np.std(returns)
        assert pytest.approx(compute_volatility(history, self.WINDOW_SIZE)["AAPL"]) == expected_vol

class TestBuildSummary:
    THRESHOLD = 0.1

    def test_no_alerts(self):
        exposures = {"AAPL": 0.05}
        changes = {"AAPL": 0.01}
        volatility = {"AAPL": 0.02}
        summary = build_summary(exposures, changes, volatility, self.THRESHOLD)
        assert "No significant alerts detected" in summary
        assert "High exposure assets" not in summary
        assert "Significant price changes" not in summary
        assert "High volatility assets" not in summary

    def test_high_exposure_alert(self):
        exposures = {"AAPL": 0.15, "MSFT": 0.05}
        changes = {"AAPL": 0.01}
        volatility = {"AAPL": 0.02}
        summary = build_summary(exposures, changes, volatility, self.THRESHOLD)
        assert "High exposure assets: AAPL" in summary
        assert "No significant alerts detected" not in summary

    def test_significant_change_alert_positive(self):
        exposures = {"AAPL": 0.05}
        changes = {"AAPL": 0.12, "MSFT": -0.05}
        volatility = {"AAPL": 0.02}
        summary = build_summary(exposures, changes, volatility, self.THRESHOLD)
        assert "Significant price changes: AAPL" in summary
        assert "No significant alerts detected" not in summary

    def test_significant_change_alert_negative(self):
        exposures = {"AAPL": 0.05}
        changes = {"AAPL": -0.12, "MSFT": 0.05}
        volatility = {"AAPL": 0.02}
        summary = build_summary(exposures, changes, volatility, self.THRESHOLD)
        assert "Significant price changes: AAPL" in summary
        assert "No significant alerts detected" not in summary
        
    def test_high_volatility_alert(self):
        exposures = {"AAPL": 0.05}
        changes = {"AAPL": 0.01}
        volatility = {"AAPL": 0.15, "MSFT": 0.08}
        summary = build_summary(exposures, changes, volatility, self.THRESHOLD)
        assert "High volatility assets: AAPL" in summary
        assert "No significant alerts detected" not in summary

    def test_multiple_alerts_sorted(self):
        exposures = {"MSFT": 0.2, "AAPL": 0.05} # MSFT high exposure
        changes = {"GOOG": 0.15, "TSLA": -0.2} # GOOG, TSLA significant change
        volatility = {"AMZN": 0.25}          # AMZN high volatility
        summary = build_summary(exposures, changes, volatility, self.THRESHOLD)
        
        assert "High exposure assets: MSFT" in summary
        # Sorted alphabetically: GOOG, TSLA
        assert "Significant price changes: GOOG, TSLA" in summary
        assert "High volatility assets: AMZN" in summary
        assert "No significant alerts detected" not in summary
        
        lines = summary.split('\n')
        assert lines[0] == "Analysis Summary:"
        assert lines[1] == "- High exposure assets: MSFT"
        assert lines[2] == "- Significant price changes: GOOG, TSLA"
        assert lines[3] == "- High volatility assets: AMZN"