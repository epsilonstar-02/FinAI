"""Calculator module for financial analysis."""
from typing import Dict, List, Any
import numpy as np
# Assuming HistoricalDataPoint is defined in models and might be type-hinted here
# For standalone use or type hinting, you might need:
# from .models import HistoricalDataPoint # If models.py is in the same directory
# Or pass it explicitly if this module doesn't know about Pydantic models

# For the context of this problem, HistoricalDataPoint comes from agents.analysis_agent.models
# but to keep calculator.py potentially more independent, we rely on structure.
# However, given main.py passes HistoricalDataPoint instances, we expect objects with .date and .close

def compute_exposures(prices: Dict[str, float]) -> Dict[str, float]:
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

def compute_changes(historical: Dict[str, List[Any]]) -> Dict[str, float]: # List[Any] to accept HistoricalDataPoint
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
        # Assuming history items have a 'date' attribute that is sortable (e.g., string 'YYYY-MM-DD')
        sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
        
        # Access 'close' attribute
        current_close = sorted_history[0].close
        previous_close = sorted_history[1].close

        if previous_close == 0: # Avoid division by zero
            changes[symbol] = 0.0
        # elif current_close == 0 and previous_close == 0: # Covered by previous_close == 0
        #     changes[symbol] = 0.0
        else:
            changes[symbol] = (current_close - previous_close) / previous_close
            
    return changes

def compute_volatility(historical: Dict[str, List[Any]], window: int) -> Dict[str, float]: # List[Any]
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
        if len(history) < 2: # Need at least 2 points to calculate 1 return
            volatility[symbol] = 0.0
            continue
            
        # Sort by date (newest first)
        sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
        
        # Take only the window size (or all if less than window)
        # The window applies to the number of data points for returns calculation,
        # so we need `window + 1` prices for `window` returns, or `window` prices for `window-1` returns.
        # The current code uses `window` prices for `window-1` returns.
        window_data = sorted_history[:min(window, len(sorted_history))]
        
        if len(window_data) < 2: # Need at least 2 points in the window to calculate returns
            volatility[symbol] = 0.0
            continue
            
        # Calculate daily returns from 'close' attribute
        prices = [point.close for point in window_data] # Prices are from newest to oldest
        
        # Returns: (P_t - P_{t-1}) / P_{t-1}. prices[i] is newer than prices[i+1]
        returns = [(prices[i] - prices[i+1]) / prices[i+1] if prices[i+1] != 0 else 0.0
                  for i in range(len(prices)-1)]
        
        if not returns: # If only one price point in window_data after filtering, or all previous prices were 0
            volatility[symbol] = 0.0
        else:
            volatility[symbol] = float(np.std(returns))
            
    return volatility

def build_summary(
    exposures: Dict[str, float], 
    changes: Dict[str, float], 
    volatility: Dict[str, float], 
    threshold: float
) -> str:
    """
    Build a summary string from analysis results, highlighting values above threshold.
    
    Args:
        exposures: Dictionary of asset symbols to exposure percentages
        changes: Dictionary of asset symbols to percentage changes
        volatility: Dictionary of asset symbols to volatility values
        threshold: Threshold for flagging high values
        
    Returns:
        Summary string with analysis and alerts
    """
    summary_lines = ["Analysis Summary:"]
    
    # Find assets with high exposure, sort for consistent output
    high_exposure_symbols = sorted([symbol for symbol, value in exposures.items() 
                                   if value > threshold])
    if high_exposure_symbols:
        summary_lines.append(f"- High exposure assets: {', '.join(high_exposure_symbols)}")
    
    # Find assets with significant changes, sort for consistent output
    significant_change_symbols = sorted([symbol for symbol, value in changes.items() 
                                        if abs(value) > threshold])
    if significant_change_symbols:
        summary_lines.append(f"- Significant price changes: {', '.join(significant_change_symbols)}")
    
    # Find assets with high volatility, sort for consistent output
    high_volatility_symbols = sorted([symbol for symbol, value in volatility.items() 
                                     if value > threshold])
    if high_volatility_symbols:
        summary_lines.append(f"- High volatility assets: {', '.join(high_volatility_symbols)}")
    
    if len(summary_lines) == 1: # Only "Analysis Summary:" is present
        summary_lines.append("- No significant alerts detected.")
    
    return "\n".join(summary_lines)