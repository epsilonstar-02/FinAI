# Enhanced Pydantic Schemas for multi-provider financial data API
from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DataProvider(str, Enum):
    """Available financial data providers"""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    FMP = "financial_modeling_prep"


# Input schema for /price endpoint with provider selection
class PriceRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z.\-]+$", description="Stock ticker symbol")
    provider: Optional[DataProvider] = Field(
        default=None,
        description="Preferred data provider (falls back to others if not available)"
    )


# Enhanced output schema for /price endpoint
class PriceResponse(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    provider: str = Field(..., description="Data source provider name")
    additional_data: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional data that may be provider-specific"
    )


# Enhanced model for a single OHLC data point
class OHLC(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None


# Enhanced output schema for /historical endpoint
class HistoricalResponse(BaseModel):
    symbol: str
    timeseries: List[OHLC]
    provider: str = Field(..., description="Data source provider name")
    start_date: date
    end_date: date
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the historical data"
    )


# Enhanced input schema for /historical endpoint
class HistoricalRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z.\-]+$")
    start: date
    end: date
    provider: Optional[DataProvider] = Field(
        default=None,
        description="Preferred data provider (falls back to others if not available)"
    )
    include_volume: bool = Field(
        default=True,
        description="Whether to include volume data"
    )


# Schema for multi-provider pricing
class MultiProviderPriceRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z.\-]+$")
    providers: List[DataProvider] = Field(
        default=[DataProvider.YAHOO_FINANCE, DataProvider.ALPHA_VANTAGE, DataProvider.FMP],
        description="List of providers to query in order of preference"
    )


# Response for multi-provider pricing
class ProviderPrice(BaseModel):
    provider: str
    price: float
    timestamp: datetime
    status: str = Field(..., description="success or error")
    error_message: Optional[str] = None


class MultiProviderPriceResponse(BaseModel):
    symbol: str
    prices: List[ProviderPrice]
    consensus_price: Optional[float] = Field(
        default=None,
        description="Consensus price across all successful providers"
    )
    timestamp: datetime
