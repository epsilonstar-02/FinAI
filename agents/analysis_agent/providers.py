"""Providers module for the Analysis Agent.

This module contains implementations of different financial analysis providers.
Each provider implements a common interface for financial calculations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
import numpy as np
import pandas as pd
from scipy import stats

class AnalysisProvider(ABC):
    """Abstract base class for analysis providers."""
    
    @abstractmethod
    def compute_exposures(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Compute portfolio exposures."""
        pass
    
    @abstractmethod
    def compute_changes(self, historical: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute price changes."""
        pass
    
    @abstractmethod
    def compute_volatility(self, historical: Dict[str, List[Any]], window: int) -> Dict[str, float]:
        """Compute volatility metrics."""
        pass
    
    @abstractmethod
    def compute_correlations(self, historical: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """Compute correlations between assets."""
        pass
    
    @abstractmethod
    def compute_risk_metrics(self, historical: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """Compute risk metrics for assets."""
        pass


class DefaultAnalysisProvider(AnalysisProvider):
    """Default implementation of analysis provider using NumPy and Pandas."""
    
    def compute_exposures(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Compute relative exposure of each asset based on current prices.
        
        Args:
            prices: Dictionary of asset symbols to current prices
            
        Returns:
            Dictionary of asset symbols to exposure percentages
        """
        total = sum(prices.values())
        if total == 0:
            return {symbol: 0.0 for symbol in prices}
        
        return {symbol: price / total for symbol, price in prices.items()}
    
    def compute_changes(self, historical: Dict[str, List[Any]]) -> Dict[str, float]:
        """
        Compute day-over-day percentage change for each asset.
        
        Args:
            historical: Dictionary of asset symbols to lists of historical data points
                      (expected to be objects with .date and .close attributes)
            
        Returns:
            Dictionary of asset symbols to percentage changes
        """
        changes = {}
        
        for symbol, history in historical.items():
            if len(history) < 2:
                changes[symbol] = 0.0
                continue
                
            # Sort by date (newest first)
            sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
            
            # Access 'close' attribute
            current_close = sorted_history[0].close
            previous_close = sorted_history[1].close
    
            if previous_close == 0:  # Avoid division by zero
                changes[symbol] = 0.0
            else:
                changes[symbol] = (current_close - previous_close) / previous_close
                
        return changes
    
    def compute_volatility(self, historical: Dict[str, List[Any]], window: int) -> Dict[str, float]:
        """
        Compute volatility (standard deviation of returns) for each asset.
        
        Args:
            historical: Dictionary of asset symbols to lists of historical data points
                      (expected to be objects with .date and .close attributes)
            window: Number of days to use for volatility calculation
            
        Returns:
            Dictionary of asset symbols to volatility values
        """
        volatility = {}
        
        for symbol, history in historical.items():
            if len(history) < 2:  # Need at least 2 points to calculate 1 return
                volatility[symbol] = 0.0
                continue
                
            # Sort by date (newest first)
            sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
            
            # Take only the window size (or all if less than window)
            window_data = sorted_history[:min(window, len(sorted_history))]
            
            if len(window_data) < 2:  # Need at least 2 points in the window to calculate returns
                volatility[symbol] = 0.0
                continue
                
            # Calculate daily returns from 'close' attribute
            prices = [point.close for point in window_data]  # Prices are from newest to oldest
            
            # Returns: (P_t - P_{t-1}) / P_{t-1}. prices[i] is newer than prices[i+1]
            returns = [(prices[i] - prices[i+1]) / prices[i+1] if prices[i+1] != 0 else 0.0
                      for i in range(len(prices)-1)]
            
            if not returns:  # If only one price point in window_data after filtering, or all previous prices were 0
                volatility[symbol] = 0.0
            else:
                volatility[symbol] = float(np.std(returns))
                
        return volatility
    
    def compute_correlations(self, historical: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compute correlation matrix between assets.
        
        Args:
            historical: Dictionary of asset symbols to lists of historical data points
            
        Returns:
            Dictionary of dictionaries with pairwise correlations
        """
        # First convert to DataFrame for easier calculation
        price_data = {}
        symbols = list(historical.keys())
        
        # Skip if we have less than 2 assets
        if len(symbols) < 2:
            return {symbol: {symbol: 1.0} for symbol in symbols}
        
        # Extract and align price data
        for symbol, history in historical.items():
            if not history:
                continue
                
            # Sort by date
            sorted_history = sorted(history, key=lambda x: x.date)
            
            # Create a series of prices
            dates = [point.date for point in sorted_history]
            prices = [point.close for point in sorted_history]
            
            if dates and prices:
                price_data[symbol] = pd.Series(prices, index=dates)
        
        # Skip if we don't have enough data
        if len(price_data) < 2:
            return {symbol: {symbol: 1.0} for symbol in price_data.keys()}
        
        # Create DataFrame from Series
        df = pd.DataFrame(price_data)
        
        # Compute returns
        returns_df = df.pct_change().dropna()
        
        # Compute correlation matrix
        corr_matrix = returns_df.corr().fillna(0).round(3)
        
        # Convert to nested dictionary
        result = {}
        for sym1 in corr_matrix.index:
            result[sym1] = {}
            for sym2 in corr_matrix.columns:
                result[sym1][sym2] = float(corr_matrix.loc[sym1, sym2])
                
        return result
    
    def compute_risk_metrics(self, historical: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compute risk metrics (Sharpe ratio, max drawdown, etc.) for each asset.
        
        Args:
            historical: Dictionary of asset symbols to lists of historical data points
            
        Returns:
            Dictionary of assets to risk metrics
        """
        risk_metrics = {}
        
        for symbol, history in historical.items():
            if len(history) < 30:  # Need reasonable amount of data
                risk_metrics[symbol] = {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "var_95": 0.0,
                    "beta": 0.0
                }
                continue
                
            # Sort by date
            sorted_history = sorted(history, key=lambda x: x.date)
            
            # Extract prices
            prices = np.array([point.close for point in sorted_history])
            
            # Calculate daily returns
            returns = np.diff(prices) / prices[:-1]
            returns = returns[~np.isnan(returns)]  # Remove NaN values
            
            if len(returns) < 5:
                risk_metrics[symbol] = {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "var_95": 0.0,
                    "beta": 0.0
                }
                continue
            
            # Calculate metrics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / running_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0.0
            
            # Beta (using a simple approximation - would need market returns for real beta)
            # Here we just use a random value between 0.5 and 1.5 for demonstration
            # In a real implementation, you would calculate this against market returns
            beta = 1.0  # Placeholder
            
            risk_metrics[symbol] = {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "var_95": float(var_95),
                "beta": float(beta)
            }
        
        return risk_metrics


class AdvancedAnalysisProvider(AnalysisProvider):
    """Advanced implementation of analysis provider with more sophisticated metrics."""
    
    def compute_exposures(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Compute portfolio exposures with sector classification."""
        # Inherits basic functionality from default provider
        provider = DefaultAnalysisProvider()
        return provider.compute_exposures(prices)
    
    def compute_changes(self, historical: Dict[str, List[Any]]) -> Dict[str, float]:
        """Compute price changes with momentum indicators."""
        changes = {}
        
        for symbol, history in historical.items():
            if len(history) < 2:
                changes[symbol] = 0.0
                continue
                
            # Sort by date (newest first)
            sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
            
            # Basic change calculation
            current_close = sorted_history[0].close
            previous_close = sorted_history[1].close
    
            if previous_close == 0:
                changes[symbol] = 0.0
            else:
                # Calculate regular change
                change = (current_close - previous_close) / previous_close
                
                # Add momentum component if we have enough data
                if len(sorted_history) >= 10:
                    prices = [point.close for point in sorted_history[:10]]
                    # Simple momentum: difference between current price and 10-day average
                    avg_price = sum(prices) / len(prices)
                    momentum = (current_close - avg_price) / avg_price if avg_price > 0 else 0
                    
                    # Blend regular change with momentum component
                    changes[symbol] = 0.7 * change + 0.3 * momentum
                else:
                    changes[symbol] = change
                
        return changes
    
    def compute_volatility(self, historical: Dict[str, List[Any]], window: int) -> Dict[str, float]:
        """Compute volatility with GARCH-like adjustment for time-varying volatility."""
        # For simplicity, we'll implement a weighted volatility calculation
        # that gives more weight to recent observations
        volatility = {}
        
        for symbol, history in historical.items():
            if len(history) < 5:  # Need more data for this approach
                volatility[symbol] = 0.0
                continue
                
            # Sort by date (newest first)
            sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
            
            # Take window size or all available
            window_data = sorted_history[:min(window, len(sorted_history))]
            
            if len(window_data) < 5:
                volatility[symbol] = 0.0
                continue
                
            # Calculate prices and returns
            prices = [point.close for point in window_data]
            returns = [(prices[i] - prices[i+1]) / prices[i+1] if prices[i+1] != 0 else 0.0
                      for i in range(len(prices)-1)]
            
            if not returns:
                volatility[symbol] = 0.0
                continue
                
            # Convert returns to numpy array for element-wise operations
            returns_array = np.array(returns)
                
            # Create exponentially decaying weights
            weights = np.exp(np.linspace(-1, 0, len(returns_array)))
            weights = weights / np.sum(weights)
            
            # Calculate weighted standard deviation
            weighted_avg = np.average(returns_array, weights=weights)
            weighted_variance = np.sum(weights * (returns_array - weighted_avg)**2)
            weighted_std = np.sqrt(weighted_variance)
            
            volatility[symbol] = float(weighted_std)
                
        return volatility
    
    def compute_correlations(self, historical: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """Compute dynamic correlations with time decay."""
        # We'll use exponentially weighted correlations
        provider = DefaultAnalysisProvider()
        return provider.compute_correlations(historical)  # Fall back to default for now
    
    def compute_risk_metrics(self, historical: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compute advanced risk metrics including conditional VaR.
        
        Includes robust error handling and NaN/inf checks for all calculations.
        """
        def safe_divide(numerator, denominator, default=0.0):
            """Safely divide two numbers, handling division by zero and invalid values."""
            if (not np.isfinite(numerator) or not np.isfinite(denominator) or 
                denominator == 0 or np.isclose(denominator, 0)):
                return default
            result = numerator / denominator
            return result if np.isfinite(result) else default
            
        risk_metrics = {}
        
        for symbol, history in historical.items():
            # Default metrics in case of insufficient data or calculation errors
            default_metrics = {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "cvar_95": 0.0,
                "calmar_ratio": 0.0
            }
            
            try:
                if not history or len(history) < 5:  # Need at least 5 data points
                    risk_metrics[symbol] = default_metrics
                    continue
                    
                # Sort by date and ensure we have valid data
                sorted_history = sorted(history, key=lambda x: x.date)
                prices = np.array([point.close for point in sorted_history if hasattr(point, 'close')])
                
                # Remove any NaN or infinite prices
                mask = np.isfinite(prices) & (prices > 0)
                if not np.any(mask):
                    risk_metrics[symbol] = default_metrics
                    continue
                    
                prices = prices[mask]
                if len(prices) < 2:  # Need at least 2 prices to calculate returns
                    risk_metrics[symbol] = default_metrics
                    continue
                
                # Calculate daily returns with safety checks
                returns = np.diff(prices) / prices[:-1]
                returns = returns[np.isfinite(returns)]
                
                if len(returns) < 5:  # Need at least 5 returns for meaningful metrics
                    risk_metrics[symbol] = default_metrics
                    continue
                
                # Calculate basic statistics with safety checks
                mean_return = np.nanmean(returns) if len(returns) > 0 else 0.0
                std_return = np.nanstd(returns, ddof=1) if len(returns) > 1 else 0.0
                
                # Sharpe ratio with safe division
                sharpe_ratio = safe_divide(mean_return, std_return)
                
                # Sortino ratio (using only downside deviation)
                downside_returns = returns[returns < 0]
                downside_std = np.nanstd(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
                sortino_ratio = safe_divide(mean_return, downside_std)
                
                # Maximum drawdown with safety checks
                try:
                    cumulative = np.cumprod(1 + returns)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdowns = np.where(running_max > 0, (running_max - cumulative) / running_max, 0.0)
                    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
                except:
                    max_drawdown = 0.0
                
                # Conditional VaR (Expected Shortfall) with safety checks
                try:
                    if len(returns) >= 5:
                        var_95_threshold = np.percentile(returns, 5, interpolation='lower')
                        cvar_losses = returns[returns <= var_95_threshold]
                        cvar_95 = np.nanmean(cvar_losses) if len(cvar_losses) > 0 else 0.0
                        cvar_95 = float(cvar_95) if np.isfinite(cvar_95) else 0.0
                    else:
                        cvar_95 = 0.0
                except:
                    cvar_95 = 0.0
                
                # Calmar ratio with safety checks
                try:
                    annual_return = (1 + mean_return) ** 252 - 1  # Annualize daily return
                    calmar_ratio = safe_divide(annual_return, max_drawdown) if max_drawdown > 1e-6 else 0.0
                except:
                    calmar_ratio = 0.0
                
                risk_metrics[symbol] = {
                    "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else 0.0,
                    "sortino_ratio": float(sortino_ratio) if np.isfinite(sortino_ratio) else 0.0,
                    "max_drawdown": float(max_drawdown) if np.isfinite(max_drawdown) else 0.0,
                    "cvar_95": cvar_95,
                    "calmar_ratio": float(calmar_ratio) if np.isfinite(calmar_ratio) else 0.0
                }
                
            except Exception as e:
                # Log the error and return safe defaults
                import logging
                logging.warning(f"Error calculating advanced risk metrics for {symbol}: {str(e)}")
                risk_metrics[symbol] = default_metrics
        
        return risk_metrics


# Provider factory
def get_provider(provider_name: str = "default") -> AnalysisProvider:
    """Get analysis provider by name."""
    providers = {
        "default": DefaultAnalysisProvider(),
        "advanced": AdvancedAnalysisProvider(),
    }
    
    return providers.get(provider_name.lower(), providers["default"])
