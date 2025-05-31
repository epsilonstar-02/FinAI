# agents/analysis_agent/providers.py
# Refined to use calculator functions and more robust risk metric calculations.

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
# scipy.stats can be used for more advanced statistical measures if needed
# from scipy import stats

from .models import HistoricalDataPoint, RiskMetrics
from .calculator import ( # Import specific functions from calculator
    compute_exposures as calc_exposures,
    compute_changes as calc_changes,
    compute_volatility as calc_volatility,
    compute_correlations_from_historical as calc_correlations
)
import logging

logger = logging.getLogger(__name__)

class AnalysisProvider(ABC):
    """Abstract base class for analysis providers."""
    
    @abstractmethod
    def compute_exposures(self, prices: Dict[str, float]) -> Dict[str, float]: pass
    
    @abstractmethod
    def compute_changes(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, float]: pass
    
    @abstractmethod
    def compute_volatility(self, historical: Dict[str, List[HistoricalDataPoint]], window: int) -> Dict[str, float]: pass
    
    @abstractmethod
    def compute_correlations(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Optional[Dict[str, Dict[str, float]]]: pass # Can return None
    
    @abstractmethod
    def compute_risk_metrics(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, Optional[RiskMetrics]]: pass


class DefaultAnalysisProvider(AnalysisProvider):
    """Default implementation using centralized calculator functions."""
    
    def compute_exposures(self, prices: Dict[str, float]) -> Dict[str, float]:
        return calc_exposures(prices)
    
    def compute_changes(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, float]:
        return calc_changes(historical)
    
    def compute_volatility(self, historical: Dict[str, List[HistoricalDataPoint]], window: int) -> Dict[str, float]:
        return calc_volatility(historical, window)
    
    def compute_correlations(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Optional[Dict[str, Dict[str, float]]]:
        return calc_correlations(historical)
    
    def _calculate_single_asset_risk_metrics(self, symbol_history: List[HistoricalDataPoint], risk_free_rate: float = 0.01) -> Optional[RiskMetrics]:
        """Helper to calculate risk metrics for a single asset."""
        if len(symbol_history) < 30: # Need at least ~30 days for somewhat meaningful daily metrics
            logger.info(f"Not enough data for {symbol_history[0].metadata.get('symbol','N/A') if symbol_history and symbol_history[0].metadata else 'N/A'} risk metrics (got {len(symbol_history)} points).")
            return None

        try:
            prices = pd.Series([p.close for p in sorted(symbol_history, key=lambda x: x.date)])
            returns = prices.pct_change().dropna()
            if len(returns) < 5: return None # Need at least a few returns

            # Sharpe Ratio (annualized, assuming 252 trading days)
            # (Mean daily return - Mean daily risk-free rate) / StdDev daily returns * sqrt(252)
            daily_rf_rate = (1 + risk_free_rate)**(1/252) - 1
            excess_returns = returns - daily_rf_rate
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else None
            
            # Max Drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else None # Positive value for drawdown

            # VaR (95%) - Value at Risk (Parametric for simplicity, assuming normal distribution)
            # Or use historical simulation: np.percentile(returns, 5)
            var_95 = np.percentile(returns, 5) if len(returns) >= 20 else None # Historical VaR

            # Beta - Placeholder as it requires market data
            beta_placeholder = 1.0 # Needs actual market index returns for real calculation

            return RiskMetrics(
                sharpe_ratio=float(sharpe_ratio) if sharpe_ratio is not None and np.isfinite(sharpe_ratio) else None,
                max_drawdown=float(max_drawdown) if max_drawdown is not None and np.isfinite(max_drawdown) else None,
                var_95=float(var_95) if var_95 is not None and np.isfinite(var_95) else None,
                beta=float(beta_placeholder) # Always returns placeholder
            )
        except Exception as e:
            logger.error(f"Error calculating default risk metrics for a symbol: {e}", exc_info=True)
            return None

    def compute_risk_metrics(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, Optional[RiskMetrics]]:
        metrics_dict: Dict[str, Optional[RiskMetrics]] = {}
        for symbol, history_list in historical.items():
            # Add symbol to metadata of each point for the helper if not already there
            for point in history_list: 
                if not hasattr(point, 'metadata') or point.metadata is None: point.metadata = {}
                point.metadata['symbol'] = symbol 
            metrics_dict[symbol] = self._calculate_single_asset_risk_metrics(history_list)
        return metrics_dict


class AdvancedAnalysisProvider(DefaultAnalysisProvider): # Inherits from Default for base calculations
    """Advanced implementation with more sophisticated metrics."""
    
    def compute_changes(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, float]:
        # Override for momentum-adjusted changes
        changes = super().compute_changes(historical) # Get basic changes
        momentum_adjusted_changes: Dict[str, float] = {}

        for symbol, basic_change in changes.items():
            history = historical.get(symbol, [])
            if len(history) >= 10: # Need enough data for 10-day SMA
                try:
                    sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
                    current_price = sorted_history[0].close
                    recent_prices = [p.close for p in sorted_history[:10]]
                    sma_10 = np.mean(recent_prices)
                    
                    momentum_factor = 0.0 # No momentum if SMA is zero or current price is zero
                    if sma_10 != 0 and current_price !=0:
                         momentum_factor = (current_price - sma_10) / sma_10
                    
                    # Blend: e.g., 70% basic change, 30% momentum
                    adjusted_change = 0.7 * basic_change + 0.3 * momentum_factor
                    momentum_adjusted_changes[symbol] = adjusted_change if np.isfinite(adjusted_change) else basic_change
                except Exception as e_mom:
                    logger.warning(f"Error calculating momentum for {symbol}: {e_mom}. Using basic change.")
                    momentum_adjusted_changes[symbol] = basic_change
            else:
                momentum_adjusted_changes[symbol] = basic_change
        return momentum_adjusted_changes

    def compute_volatility(self, historical: Dict[str, List[HistoricalDataPoint]], window: int) -> Dict[str, float]:
        # Override for exponentially weighted volatility
        volatilities: Dict[str, float] = {}
        if not historical: return volatilities

        for symbol, history in historical.items():
            if not history or len(history) < 5: # Need a few points for EWMA
                volatilities[symbol] = 0.0
                continue
            
            try:
                sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
                prices_for_returns = [p.close for p in sorted_history[:min(window + 1, len(sorted_history))]]
                if len(prices_for_returns) < 2:
                    volatilities[symbol] = 0.0
                    continue
                
                returns = pd.Series([
                    (prices_for_returns[i] - prices_for_returns[i+1]) / prices_for_returns[i+1]
                    if prices_for_returns[i+1] != 0 else 0.0
                    for i in range(len(prices_for_returns)-1)
                ])

                if returns.empty:
                    volatilities[symbol] = 0.0
                    continue

                # Exponentially weighted moving standard deviation
                # Span correlates to window, e.g., span of `window` days for returns
                ewm_std = returns.ewm(span=max(2, window-1), adjust=True).std().iloc[-1] # Use last value
                volatilities[symbol] = float(ewm_std) if pd.notnull(ewm_std) and np.isfinite(ewm_std) else 0.0

            except Exception as e_vol:
                logger.warning(f"Error calculating EWMA volatility for {symbol}: {e_vol}. Using default calc.")
                # Fallback to default volatility calculation for this symbol
                volatilities[symbol] = super().compute_volatility({symbol: history}, window).get(symbol, 0.0)
                
        return volatilities

    def _calculate_single_asset_advanced_risk_metrics(self, symbol_history: List[HistoricalDataPoint], risk_free_rate: float = 0.01) -> Optional[RiskMetrics]:
        """Calculates advanced risk metrics including Sortino, Calmar, CVaR."""
        base_metrics = super()._calculate_single_asset_risk_metrics(symbol_history, risk_free_rate)
        if base_metrics is None: return None # Not enough data or base calc failed

        try:
            prices = pd.Series([p.close for p in sorted(symbol_history, key=lambda x: x.date)])
            returns = prices.pct_change().dropna()
            if len(returns) < 20: # More data for stable advanced metrics
                 logger.info(f"Not enough returns for advanced risk metrics for {symbol_history[0].metadata.get('symbol','N/A') if symbol_history else 'N/A'} (got {len(returns)} returns).")
                 return base_metrics # Return base if not enough for advanced

            daily_rf_rate = (1 + risk_free_rate)**(1/252) - 1
            
            # Sortino Ratio
            downside_returns = returns[returns < daily_rf_rate] # Returns below risk-free rate
            expected_return = np.mean(returns)
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
            sortino_ratio = ((expected_return - daily_rf_rate) / downside_std) * np.sqrt(252) if downside_std != 0 else None
            
            # CVaR (95%) - Conditional Value at Risk (Expected Shortfall)
            var_95_val = np.percentile(returns, 5) # Historical VaR
            cvar_95 = np.mean(returns[returns <= var_95_val]) if len(returns[returns <= var_95_val]) > 0 else None

            # Calmar Ratio (Annualized Return / Max Drawdown)
            annualized_return = (1 + np.mean(returns))**252 - 1
            max_dd = base_metrics.max_drawdown if base_metrics.max_drawdown is not None and base_metrics.max_drawdown > 1e-6 else None # Avoid div by zero
            calmar_ratio = annualized_return / max_dd if max_dd is not None else None

            # Update base_metrics with advanced ones
            base_metrics.sortino_ratio = float(sortino_ratio) if sortino_ratio is not None and np.isfinite(sortino_ratio) else None
            base_metrics.cvar_95 = float(cvar_95) if cvar_95 is not None and np.isfinite(cvar_95) else None
            base_metrics.calmar_ratio = float(calmar_ratio) if calmar_ratio is not None and np.isfinite(calmar_ratio) else None
            return base_metrics

        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics for a symbol: {e}", exc_info=True)
            return base_metrics # Return base if advanced calc fails

    def compute_risk_metrics(self, historical: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, Optional[RiskMetrics]]:
        metrics_dict: Dict[str, Optional[RiskMetrics]] = {}
        for symbol, history_list in historical.items():
            for point in history_list: # Ensure metadata for logging in helper
                if not hasattr(point, 'metadata') or point.metadata is None: point.metadata = {}
                point.metadata['symbol'] = symbol
            metrics_dict[symbol] = self._calculate_single_asset_advanced_risk_metrics(history_list)
        return metrics_dict


# Provider factory
_provider_instances: Dict[str, AnalysisProvider] = {}

def get_provider(provider_name: str = "default") -> AnalysisProvider:
    """Cached factory for analysis providers."""
    provider_key = provider_name.lower()
    if provider_key not in _provider_instances:
        if provider_key == "default":
            _provider_instances[provider_key] = DefaultAnalysisProvider()
        elif provider_key == "advanced":
            _provider_instances[provider_key] = AdvancedAnalysisProvider()
        else:
            logger.warning(f"Unknown provider name: {provider_name}. Falling back to default.")
            _provider_instances[provider_key] = DefaultAnalysisProvider() # Fallback
    return _provider_instances[provider_key]