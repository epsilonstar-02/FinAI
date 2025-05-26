#Pydantic Schemas
from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import List

# Input schema for /price endpoint
class PriceRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5, regex="^[A-Z]+$")

# Output schema for /price endpoint
class PriceResponse(BaseModel):
    symbol: str
    price: float
    timestamp: datetime

# Model for a single OHLC data point
class OHLC(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float

# Output schema for /historical endpoint
class HistoricalResponse(BaseModel):
    symbol: str
    timeseries: List[OHLC]

# Input schema for /historical endpoint
class HistoricalRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5, regex="^[A-Z]+$")
    start: date
    end: date