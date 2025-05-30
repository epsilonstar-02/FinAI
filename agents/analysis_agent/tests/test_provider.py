import pytest
import numpy as np
import pandas as pd
from typing import Dict, List

# Adjust import paths as necessary
from agents.analysis_agent.providers import (
    get_provider,
    DefaultAnalysisProvider,
    AdvancedAnalysisProvider,
    AnalysisProvider
)
from agents.analysis_agent.models import HistoricalDataPoint, RiskMetrics

# Helper to create HistoricalDataPoint objects easily
def HDP(date: str, close: float) -> HistoricalDataPoint:
    return HistoricalDataPoint(date=date, close=close)

@pytest.fixture
def default_provider() -> DefaultAnalysisProvider:
    return DefaultAnalysisProvider()

@pytest.fixture
def advanced_provider() -> AdvancedAnalysisProvider:
    return AdvancedAnalysisProvider()

@pytest.fixture
def historical_data_fixture() -> Dict[str, List[HistoricalDataPoint]]:
    return {
        "AAPL": [
            HDP("2023-01-05", 105.0), HDP("2023-01-04", 103.0), HDP("2023-01-03", 104.0),
            HDP("2023-01-02", 102.0), HDP("2023-01-01", 100.0)
        ],
        "MSFT": [ # Data for MSFT to calculate correlations
            HDP("2023-01-05", 208.0), HDP("2023-01-04", 205.0), HDP("2023-01-03", 206.0),
            HDP("2023-01-02", 202.0), HDP("2023-01-01", 200.0)
        ]
    }

@pytest.fixture
def long_historical_data_fixture() -> Dict[str, List[HistoricalDataPoint]]:
    # Generate 60 days of data for risk metrics
    data = {}
    for symbol in ["SYM1", "SYM2"]:
        history = []
        price = 100.0
        for i in range(60):
            date_str = f"2023-03-{str(i+1).zfill(2)}" # Create unique dates like 2023-03-01, 2023-03-02 etc.
            if i >= 31: # Make dates for March
                 date_str = f"2023-03-{str(i-30).zfill(2)}"
            elif i >= 0 : # Make dates for Feb (assuming 28 days)
                 date_str = f"2023-02-{str(i+1).zfill(2)}"

            history.append(HDP(date_str, price + np.random.uniform(-1,1))) # Add some noise
            price += np.random.uniform(-0.5,0.5) # Slight trend
        data[symbol] = sorted(history, key=lambda x: x.date, reverse=True) # Newest first for consistency
    return data


class TestProviderFactory:
    def test_get_default_provider(self):
        provider = get_provider("default")
        assert isinstance(provider, DefaultAnalysisProvider)
        provider = get_provider("DEFAULT") # Case-insensitivity
        assert isinstance(provider, DefaultAnalysisProvider)

    def test_get_advanced_provider(self):
        provider = get_provider("advanced")
        assert isinstance(provider, AdvancedAnalysisProvider)
        provider = get_provider("ADVANCED")
        assert isinstance(provider, AdvancedAnalysisProvider)

    def test_get_unknown_provider_falls_back_to_default(self):
        provider = get_provider("unknown_provider_xyz")
        assert isinstance(provider, DefaultAnalysisProvider)

    def test_get_provider_no_arg(self): # Default argument value
        provider = get_provider()
        assert isinstance(provider, DefaultAnalysisProvider)

class TestDefaultAnalysisProvider:
    # compute_exposures, compute_changes, compute_volatility are largely
    # direct calls to calculator functions. They are tested more thoroughly
    # in test_calculator.py. Here, we do a basic check.
    def test_compute_exposures(self, default_provider: DefaultAnalysisProvider):
        prices = {"A": 60, "B": 40}
        expected = {"A": 0.6, "B": 0.4}
        assert default_provider.compute_exposures(prices) == expected

    def test_compute_changes(self, default_provider: DefaultAnalysisProvider, historical_data_fixture):
        # AAPL: Newest=105 (01-05), Prev=103 (01-04)
        # (105-103)/103 = 2/103
        expected_aapl_change = (105.0 - 103.0) / 103.0
        changes = default_provider.compute_changes(historical_data_fixture)
        assert pytest.approx(changes["AAPL"]) == expected_aapl_change

    def test_compute_volatility(self, default_provider: DefaultAnalysisProvider, historical_data_fixture):
        window = 3
        # AAPL data: [105, 103, 104, 102, 100] (newest to oldest in fixture)
        # Provider sorts it again by date, reverse=True: [105, 103, 104, 102, 100]
        # Window_data (prices newest to oldest for window=3): [105, 103, 104]
        # Returns: r1=(105-103)/103, r2=(103-104)/104
        returns_aapl = [(105.0-103.0)/103.0, (103.0-104.0)/104.0]
        expected_aapl_vol = np.std(returns_aapl)
        
        vol = default_provider.compute_volatility(historical_data_fixture, window)
        assert pytest.approx(vol["AAPL"]) == expected_aapl_vol

    def test_compute_correlations(self, default_provider: DefaultAnalysisProvider, historical_data_fixture):
        correlations = default_provider.compute_correlations(historical_data_fixture)
        
        assert "AAPL" in correlations
        assert "MSFT" in correlations["AAPL"]
        assert correlations["AAPL"]["AAPL"] == 1.0
        assert correlations["MSFT"]["MSFT"] == 1.0
        
        # Check symmetry
        assert pytest.approx(correlations["AAPL"]["MSFT"]) == pytest.approx(correlations["MSFT"]["AAPL"])
        
        # For the given data, prices move roughly together, so expect positive correlation
        assert correlations["AAPL"]["MSFT"] > 0 

        # Test with one asset
        single_asset_data = {"AAPL": historical_data_fixture["AAPL"]}
        single_corr = default_provider.compute_correlations(single_asset_data)
        assert single_corr == {"AAPL": {"AAPL": 1.0}}

        # Test with no common dates (should result in NaN -> 0)
        no_common_dates_data = {
            "SYM1": [HDP("2023-01-01", 100), HDP("2023-01-02", 101)],
            "SYM2": [HDP("2023-02-01", 200), HDP("2023-02-02", 201)],
        }
        corr_no_common = default_provider.compute_correlations(no_common_dates_data)
        # If pct_change().dropna() leads to empty DFs or DFs with no common index for corr,
        # pandas fillna(0) handles it.
        assert corr_no_common["SYM1"]["SYM2"] == 0.0

    def test_compute_risk_metrics(self, default_provider: DefaultAnalysisProvider, long_historical_data_fixture):
        metrics = default_provider.compute_risk_metrics(long_historical_data_fixture)
        assert "SYM1" in metrics
        assert "SYM2" in metrics
        
        for symbol in ["SYM1", "SYM2"]:
            sym_metrics = metrics[symbol]
            # These are the keys DefaultAnalysisProvider is expected to return
            expected_keys = ["sharpe_ratio", "max_drawdown", "var_95", "beta"]
            for k in expected_keys:
                assert k in sym_metrics
            assert isinstance(sym_metrics["sharpe_ratio"], float)
            assert isinstance(sym_metrics["max_drawdown"], float)
            assert 0.0 <= sym_metrics["max_drawdown"] <= 1.0
            assert isinstance(sym_metrics["var_95"], float)
            assert sym_metrics["var_95"] <= 0.0 # VaR is typically a loss
            assert sym_metrics["beta"] == 1.0 # Placeholder value

    def test_compute_risk_metrics_insufficient_data(self, default_provider: DefaultAnalysisProvider, historical_data_fixture):
        # historical_data_fixture has only 5 points, Default provider needs > 30
        metrics = default_provider.compute_risk_metrics(historical_data_fixture)
        for symbol in historical_data_fixture.keys():
            assert metrics[symbol] == {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "beta": 0.0
            }

class TestAdvancedAnalysisProvider:
    # Exposures are inherited from Default, no need to re-test deeply
    def test_compute_exposures(self, advanced_provider: AdvancedAnalysisProvider):
        prices = {"A": 60, "B": 40}
        expected = {"A": 0.6, "B": 0.4}
        assert advanced_provider.compute_exposures(prices) == expected

    def test_compute_changes_momentum(self, advanced_provider: AdvancedAnalysisProvider):
        # Need at least 10 data points for momentum component
        history_long = [HDP(f"2023-01-{day:02d}", 100.0 + day) for day in range(1, 12)] # 11 points
        # Newest is 2023-01-11 (110.0), Prev is 2023-01-10 (109.0)
        # Current close = 110.0, Previous close = 109.0
        # Regular change = (110-109)/109 = 1/109 approx 0.00917
        
        # Momentum part: 10-day avg of [110, 109, ..., 101]
        # Prices for avg: 110, 109, 108, 107, 106, 105, 104, 103, 102, 101
        # Sum = 1055, Avg = 105.5
        # Momentum = (110 - 105.5) / 105.5 = 4.5 / 105.5 approx 0.04265
        
        # Blended = 0.7 * (1/109) + 0.3 * (4.5/105.5)
        #         = 0.7 * 0.0091743 + 0.3 * 0.0426539
        #         = 0.006422 + 0.012796
        #         = 0.019218
        
        data = {"AAPL": list(reversed(history_long))} # Provider expects newest first in list for its sorting logic if any.
                                                      # Or rather, it sorts by date. Order in list doesn't matter.
        
        changes = advanced_provider.compute_changes({"AAPL": history_long})
        
        current_close = 100.0 + 11
        previous_close = 100.0 + 10
        regular_change = (current_close - previous_close) / previous_close
        
        prices_for_avg = [100.0 + day for day in range(2,12)] # Prices from day 2 to day 11
        avg_price = sum(prices_for_avg) / len(prices_for_avg)
        momentum = (current_close - avg_price) / avg_price
        
        expected_change = 0.7 * regular_change + 0.3 * momentum
        assert pytest.approx(changes["AAPL"]) == expected_change

    def test_compute_changes_less_than_10_points(self, advanced_provider: AdvancedAnalysisProvider, historical_data_fixture):
        # AAPL data has 5 points, so no momentum component
        # Newest=105 (01-05), Prev=103 (01-04)
        # Change = (105-103)/103
        expected_aapl_change = (105.0 - 103.0) / 103.0
        changes = advanced_provider.compute_changes(historical_data_fixture)
        assert pytest.approx(changes["AAPL"]) == expected_aapl_change


    def test_compute_volatility_weighted(self, advanced_provider: AdvancedAnalysisProvider, historical_data_fixture):
        # Case 1: len(window_data) < 5, provider should return 0.0
        window_case1 = 3
        # Provider logic: historical_data_fixture["AAPL"] has 5 points.
        # window_data will have min(3, 5) = 3 points.
        # Condition `if len(window_data) < 5:` (i.e., `3 < 5`) is true.
        # So, volatility[symbol] = 0.0
        expected_vol_case1 = 0.0

        vol_case1 = advanced_provider.compute_volatility(historical_data_fixture, window_case1)
        assert vol_case1["AAPL"] == pytest.approx(expected_vol_case1)

        # Case 2: Test actual weighted calculation path with enough data in window
        window_case2 = 5 # Ensure len(window_data) is not < 5
        # Create history with 5 points for AAPL
        history_for_calc = {
            "AAPL": [HDP(f"2023-01-{i+1:02d}", 100.0 + i*2) for i in range(window_case2)]
        }
        # Provider sorts by date, newest first.
        # If HDP creates dates like 01, 02, 03, 04, 05:
        # Prices after sorting (newest to oldest): [108, 106, 104, 102, 100]
        
        # Manually calculate expected_vol for these 5 prices
        # Prices from newest to oldest:
        prices_in_window = [100.0 + (window_case2 - 1 - i) * 2 for i in range(window_case2)] # [108, 106, 104, 102, 100]
        
        returns_for_calc = np.array([
            (prices_in_window[i] - prices_in_window[i+1]) / prices_in_window[i+1]
            for i in range(len(prices_in_window)-1)
        ]) # Should have 4 returns
        
        weights_for_calc = np.exp(np.linspace(-1, 0, len(returns_for_calc)))
        weights_for_calc = weights_for_calc / np.sum(weights_for_calc)
        
        avg_ret_for_calc = np.average(returns_for_calc, weights=weights_for_calc)
        variance_for_calc = np.sum(weights_for_calc * (returns_for_calc - avg_ret_for_calc)**2)
        expected_vol_case2_nonzero = np.sqrt(variance_for_calc)

        vol_case2 = advanced_provider.compute_volatility(history_for_calc, window_case2)
        assert vol_case2["AAPL"] == pytest.approx(expected_vol_case2_nonzero)
        assert vol_case2["AAPL"] > 0 # Ensure it's not zero

    def test_compute_volatility_insufficient_data(self, advanced_provider: AdvancedAnalysisProvider):
        # Advanced provider needs len(history) < 5 or len(window_data) < 5
        short_history = {"AAPL": [HDP("2023-01-02", 101), HDP("2023-01-01", 100)]} # 2 points
        vol = advanced_provider.compute_volatility(short_history, 3)
        assert vol["AAPL"] == 0.0

        four_points_history = {"AAPL": [HDP(f"2023-01-0{i}", 100+i) for i in range(4,0,-1)]} # 4 points
        vol_4pts = advanced_provider.compute_volatility(four_points_history, 5) # window > data length
        assert vol_4pts["AAPL"] == 0.0 # as len(window_data) will be 4, which is < 5

    def test_compute_correlations_fallback(self, advanced_provider: AdvancedAnalysisProvider, default_provider: DefaultAnalysisProvider, historical_data_fixture):
        # Advanced provider currently uses Default's correlation
        corr_advanced = advanced_provider.compute_correlations(historical_data_fixture)
        corr_default = default_provider.compute_correlations(historical_data_fixture)
        assert corr_advanced == corr_default

    def test_compute_risk_metrics_advanced(self, advanced_provider: AdvancedAnalysisProvider, long_historical_data_fixture):
        metrics = advanced_provider.compute_risk_metrics(long_historical_data_fixture)
        assert "SYM1" in metrics
        
        for symbol in ["SYM1"]:
            sym_metrics = metrics[symbol]
            # These are the keys AdvancedAnalysisProvider is expected to return
            expected_keys = ["sharpe_ratio", "sortino_ratio", "max_drawdown", "cvar_95", "calmar_ratio"]
            for k in expected_keys:
                assert k in sym_metrics
            
            assert isinstance(sym_metrics["sharpe_ratio"], float)
            assert isinstance(sym_metrics["sortino_ratio"], float)
            assert isinstance(sym_metrics["max_drawdown"], float)
            assert 0.0 <= sym_metrics["max_drawdown"] <= 1.0
            assert isinstance(sym_metrics["cvar_95"], float)
            assert sym_metrics["cvar_95"] <= 0.0 # CVaR is a loss
            assert isinstance(sym_metrics["calmar_ratio"], float)

            # Check for keys NOT expected from Advanced provider, which ARE in RiskMetrics model
            # This highlights the potential Pydantic validation issue.
            assert "beta" not in sym_metrics 
            assert "var_95" not in sym_metrics

    def test_compute_risk_metrics_advanced_insufficient_data(self, advanced_provider: AdvancedAnalysisProvider, historical_data_fixture):
        # historical_data_fixture has only 5 points, Advanced provider needs > 60
        metrics = advanced_provider.compute_risk_metrics(historical_data_fixture)
        for symbol in historical_data_fixture.keys():
            assert metrics[symbol] == { # Default values for insufficient data
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "cvar_95": 0.0,
                "calmar_ratio": 0.0
            }