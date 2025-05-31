# agents/analysis_agent/calculator.py
# Refined calculations, type hints, and robustness.

from typing import Dict, List, Any, Optional # Added Optional
import numpy as np
import pandas as pd # For correlation convenience
from .models import HistoricalDataPoint # Use specific model for type hint

logger = logging.getLogger(__name__) # Added logger


def compute_exposures(prices: Dict[str, float]) -> Dict[str, float]:
    if not prices: return {}
    total_value = sum(prices.values())
    if total_value == 0:
        # Avoid division by zero; if all prices are 0, exposure is undefined or equally 0.
        return {symbol: 0.0 for symbol in prices}
    return {symbol: price / total_value for symbol, price in prices.items()}


def compute_changes(historical_data: Dict[str, List[HistoricalDataPoint]]) -> Dict[str, float]:
    changes: Dict[str, float] = {}
    if not historical_data: return changes

    for symbol, history in historical_data.items():
        if not history or len(history) < 2:
            changes[symbol] = 0.0 # Not enough data for change
            continue
        
        # Sort by date, most recent first. Assuming date strings are comparable (YYYY-MM-DD).
        try:
            sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
        except (TypeError, ValueError) as e_sort:
            logger.warning(f"Could not sort history for {symbol} due to date format issues: {e_sort}. Skipping change calc.")
            changes[symbol] = 0.0
            continue
            
        current_price = sorted_history[0].close
        previous_price = sorted_history[1].close

        if previous_price == 0: # Avoid division by zero
            # If prev price is 0, change is undefined or infinite if current is non-zero.
            # Assign a large number or 0.0 based on desired behavior. For now, 0.0.
            changes[symbol] = 0.0 if current_price == 0 else np.inf # Or handle as error/None
            if np.isinf(changes[symbol]): logger.warning(f"Infinite change for {symbol} (prev_price=0, current_price={current_price})")
        else:
            changes[symbol] = (current_price - previous_price) / previous_price
            
    return changes


def compute_volatility(historical_data: Dict[str, List[HistoricalDataPoint]], window: int) -> Dict[str, float]:
    volatilities: Dict[str, float] = {}
    if not historical_data: return volatilities

    for symbol, history in historical_data.items():
        if not history or len(history) < 2: # Need at least 2 data points for 1 return
            volatilities[symbol] = 0.0
            continue

        try:
            sorted_history = sorted(history, key=lambda x: x.date, reverse=True)
        except (TypeError, ValueError) as e_sort:
            logger.warning(f"Could not sort history for {symbol} (volatility): {e_sort}. Skipping.")
            volatilities[symbol] = 0.0
            continue
            
        # We need `window` returns, so `window + 1` prices. Or, `window` prices for `window - 1` returns.
        # If window is 10, use 10 most recent prices to calculate 9 returns.
        prices_for_returns = [p.close for p in sorted_history[:min(window + 1, len(sorted_history))]]
        
        if len(prices_for_returns) < 2:
            volatilities[symbol] = 0.0
            continue
            
        # Prices are newest to oldest. Returns: (P_t - P_{t-1}) / P_{t-1} where P_t is prices[i], P_{t-1} is prices[i+1]
        returns = np.array([
            (prices_for_returns[i] - prices_for_returns[i+1]) / prices_for_returns[i+1]
            if prices_for_returns[i+1] != 0 else 0.0
            for i in range(len(prices_for_returns)-1)
        ])
        
        if returns.size == 0: # No returns calculated (e.g., only 1 price point after filtering)
            volatilities[symbol] = 0.0
        else:
            # Annualized volatility: std(daily_returns) * sqrt(252)
            # Here we return daily std dev. Annualization can be done by caller if needed.
            vol = float(np.std(returns))
            volatilities[symbol] = vol if np.isfinite(vol) else 0.0 # Handle NaN/inf from std if all returns are same/problematic
            
    return volatilities


def compute_correlations_from_historical(historical_data: Dict[str, List[HistoricalDataPoint]]) -> Optional[Dict[str, Dict[str, float]]]:
    """Computes correlation matrix from historical closing prices."""
    if not historical_data or len(historical_data) < 2:
        return None # Not enough assets for correlation

    # Create a DataFrame with dates as index and symbols as columns, filled with closing prices
    all_dates = sorted(list(set(point.date for history_list in historical_data.values() for point in history_list)))
    price_df_data = {symbol: {point.date: point.close for point in history} for symbol, history in historical_data.items()}
    
    price_df = pd.DataFrame(index=all_dates)
    for symbol, date_price_map in price_df_data.items():
        price_df[symbol] = price_df.index.map(date_price_map)

    price_df = price_df.dropna(axis=1, how='all') # Drop symbols with no price data
    price_df = price_df.fillna(method='ffill').fillna(method='bfill') # Fill missing values
    
    if price_df.shape[1] < 2: # Need at least two valid series for correlation
        return None

    returns_df = price_df.pct_change().dropna(how='all') # Calculate returns and drop rows with all NaNs
    
    if returns_df.empty or returns_df.shape[0] < 2 : # Need at least 2 return periods
        return None

    correlation_matrix = returns_df.corr().fillna(0.0) # Fill NaN correlations with 0
    
    # Convert to nested dict, ensuring float values
    return {col: correlation_matrix[col].apply(lambda x: float(x) if pd.notnull(x) else 0.0).to_dict() 
            for col in correlation_matrix.columns}


def build_summary(
    exposures: Dict[str, float], 
    changes: Dict[str, float], 
    volatility: Dict[str, float], 
    threshold: float = 0.05 # Use setting from config in main.py
) -> str:
    summary_parts = ["Financial Analysis Summary:"]
    alerts_found = False

    # Exposures
    if exposures:
        sorted_exposures = sorted(exposures.items(), key=lambda item: item[1], reverse=True)
        top_exposure_str = ", ".join([f"{sym} ({val:.2%})" for sym, val in sorted_exposures[:3]])
        summary_parts.append(f"- Top Exposures: {top_exposure_str}")
        high_exp = [sym for sym, val in sorted_exposures if val > threshold + 0.15] # Higher threshold for "high" exposure
        if high_exp: summary_parts.append(f"  - Notably High Exposure: {', '.join(high_exp)}"); alerts_found = True
    
    # Changes
    if changes:
        # Sort by absolute change to find most significant, then format with original sign
        sorted_changes = sorted(changes.items(), key=lambda item: abs(item[1]), reverse=True)
        significant_changes_str = ", ".join([f"{sym} ({val:+.2%})" for sym, val in sorted_changes if abs(val) > threshold])
        if significant_changes_str:
            summary_parts.append(f"- Significant Price Changes (> {threshold:.0%}): {significant_changes_str}")
            alerts_found = True
        else:
            summary_parts.append(f"- No significant price changes detected above {threshold:.0%}.")

    # Volatility
    if volatility:
        sorted_volatility = sorted(volatility.items(), key=lambda item: item[1], reverse=True)
        top_vol_str = ", ".join([f"{sym} ({val:.3f})" for sym, val in sorted_volatility[:3]]) # Daily std dev
        summary_parts.append(f"- Top Volatilities (Daily Std Dev): {top_vol_str}")
        high_vol = [sym for sym, val in sorted_volatility if val > threshold] # Using original threshold
        if high_vol: summary_parts.append(f"  - Notably High Volatility: {', '.join(high_vol)}"); alerts_found = True

    if not alerts_found and len(summary_parts) > 1: # If we added sections but no specific alerts
        summary_parts.append("- Overall market conditions appear stable within observed metrics.")
    elif not alerts_found and len(summary_parts) == 1: # No data processed
        summary_parts.append("- Insufficient data for a detailed summary.")
        
    return "\n".join(summary_parts)