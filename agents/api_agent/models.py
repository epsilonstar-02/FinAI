# agents/api_agent/models.py
# No significant changes needed, models are well-defined.
# Added Config.validate_assignment = True to OHLC for stricter validation.
# Corrected validator usage from Pydantic v1 to v2 field_validator for OHLC.

from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator # validator removed, using field_validator
from typing import List, Optional, Dict, Any
from .config import DataProvider


class PriceResponse(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    provider: str = Field(..., description="Data source provider name (e.g., 'yahoo_finance')")
    additional_data: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional data that may be provider-specific"
    )

    @field_validator('price')
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Price must be non-negative')
        return v


class OHLC(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    
    model_config = { # Pydantic v2 style config
        "validate_assignment": True
    }

    @field_validator('open', 'high', 'low', 'close')
    @classmethod
    def prices_must_be_positive(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f'{info.field_name} must be non-negative')
        return v

    @field_validator('volume')
    @classmethod
    def volume_must_be_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError('Volume must be non-negative')
        return v
    
    @field_validator('high')
    @classmethod
    def high_must_be_gte_others(cls, v: float, info) -> float:
        values = info.data # Access other field values via info.data
        if 'low' in values and v < values['low']:
            raise ValueError('High price must be greater than or equal to low price.')
        if 'open' in values and v < values['open']:
             raise ValueError('High price must be greater than or equal to open price.')
        if 'close' in values and v < values['close']:
             raise ValueError('High price must be greater than or equal to close price.')
        return v

    @field_validator('low')
    @classmethod
    def low_must_be_lte_others(cls, v: float, info) -> float:
        values = info.data
        # No need to check against high if high_must_be_gte_others is already there
        # but let's keep for explicit validation for 'low' itself.
        if 'high' in values and v > values['high']:
             raise ValueError('Low price must be less than or equal to high price.')
        if 'open' in values and v > values['open']:
             raise ValueError('Low price must be less than or equal to open price.')
        if 'close' in values and v > values['close']:
             raise ValueError('Low price must be less than or equal to close price.')
        return v


class HistoricalResponse(BaseModel):
    symbol: str
    timeseries: List[OHLC]
    provider: str = Field(..., description="Data source provider name")
    start_date: date = Field(..., description="Actual start date of the returned timeseries data")
    end_date: date = Field(..., description="Actual end date of the returned timeseries data")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the historical data (e.g., currency, time_zone)"
    )


class MultiProviderPriceRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z0-9.\-]+$", description="Stock ticker symbol")
    providers: List[DataProvider] = Field(
        default_factory=lambda: list(DataProvider), # Use default_factory for mutable defaults
        min_length=1,
        description="List of providers to query. Enum values expected."
    )


class ProviderPrice(BaseModel):
    provider: str 
    price: Optional[float] = None
    timestamp: datetime 
    status: str = Field(..., description="'success' or 'error'")
    error_message: Optional[str] = None

    @field_validator('price')
    @classmethod
    def price_must_be_positive_if_not_none(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError('Price must be non-negative if provided')
        return v


class MultiProviderPriceResponse(BaseModel):
    symbol: str
    prices: List[ProviderPrice]
    consensus_price: Optional[float] = Field(
        default=None,
        description="Consensus price (e.g., median) across all successful provider responses. Null if no successful responses or only one."
    )
    timestamp: datetime

    @field_validator('consensus_price')
    @classmethod
    def consensus_price_must_be_positive_if_not_none(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError('Consensus price must be non-negative if provided')
        return v