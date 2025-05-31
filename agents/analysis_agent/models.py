# agents/analysis_agent/models.py
# Use field_validator for Pydantic v2.
# Add model_config to relevant models.

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator # validator removed
from datetime import datetime # Added for potential use in future metadata

class HistoricalDataPoint(BaseModel):
    date: str = Field(..., description="Date in ISO format (YYYY-MM-DD)") # Could be pydantic.AwareDatetime or date
    close: float = Field(..., description="Closing price", gt=0) # Price should be > 0
    open: Optional[float] = Field(None, description="Opening price", gt=0)
    high: Optional[float] = Field(None, description="High price", gt=0)
    low: Optional[float] = Field(None, description="Low price", gt=0)
    volume: Optional[int] = Field(None, description="Trading volume", ge=0) # Volume >= 0

    # model_config can be added for alias_generator or other settings if needed
    model_config = {"extra": "allow"} # Allow extra fields if API returns more

    @field_validator('high')
    @classmethod
    def high_gte_low(cls, v: Optional[float], info) -> Optional[float]:
        # Pydantic v2: info.data contains validated fields so far
        low_price = info.data.get('low')
        if v is not None and low_price is not None and v < low_price:
            raise ValueError('High price must be greater than or equal to low price')
        return v

    @field_validator('low')
    @classmethod
    def low_lte_others(cls, v: Optional[float], info) -> Optional[float]:
        # More comprehensive checks for low relative to open/close/high
        open_price = info.data.get('open')
        close_price = info.data.get('close') # close is required
        high_price = info.data.get('high') # high check is somewhat redundant due to high_gte_low

        if v is not None:
            if open_price is not None and v > open_price:
                # This can happen (e.g. gap down, low is still above prev open if it's for current day's low)
                # logger.warning("Low price is greater than open price.") # Informational
                pass
            if v > close_price: # Close is required field
                # logger.warning("Low price is greater than close price.") # Informational
                pass
            if high_price is not None and v > high_price: # This should be caught by high_gte_low
                raise ValueError('Low price must be less than or equal to high price')
        return v


class RiskMetrics(BaseModel):
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio") # Made Optional, as calc can fail
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown percentage (0.0 to 1.0)")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence, negative value)")
    beta: Optional[float] = Field(None, description="Beta (market sensitivity)")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    cvar_95: Optional[float] = Field(None, description="Conditional VaR (95%, negative value)")

class ProviderInfo(BaseModel):
    name: str
    version: str = "1.0.0" # Default version
    fallback_used: bool = False
    execution_time_ms: Optional[float] = None

class AnalyzeRequest(BaseModel):
    prices: Dict[str, float] = Field(..., description="Current prices for assets (symbol: price)")
    historical: Dict[str, List[HistoricalDataPoint]] = Field(
        default_factory=dict, # Allow empty historical data
        description="Historical price data for assets (symbol: List[HistoricalDataPoint])"
    )
    provider: Optional[str] = None
    include_correlations: bool = False # Default to False for leaner requests
    include_risk_metrics: bool = False # Default to False

    model_config = {"extra": "forbid"} # Forbid extra fields in request

    @field_validator('prices')
    @classmethod
    def prices_must_be_positive_in_request(cls, v: Dict[str, float]) -> Dict[str, float]:
        for symbol, price in v.items():
            if price <= 0:
                raise ValueError(f"Price for symbol '{symbol}' must be positive, got {price}")
        return v


class AnalyzeResponse(BaseModel):
    exposures: Dict[str, float]
    changes: Dict[str, float]
    volatility: Dict[str, float]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    risk_metrics: Optional[Dict[str, Optional[RiskMetrics]]] = None # Values can be None if a symbol fails risk calc
    summary: str
    provider_info: ProviderInfo
    
    # Example data moved to main.py openapi_extra for cleaner models file
    model_config = {"extra": "ignore"} # Ignore extra fields from provider calculations if any


class ErrorResponse(BaseModel):
    status: str = Field("error", description="Status of the response")
    message: str
    details: Optional[Any] = None

    model_config = {"extra": "ignore"}