"""Models for the Analysis Agent."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class HistoricalDataPoint(BaseModel):
    """Model for a single historical data point."""
    date: str # Should be ISO format string for proper sorting, e.g., YYYY-MM-DD
    close: float

class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint."""
    prices: Dict[str, float] = Field(..., description="Current prices for assets")
    historical: Dict[str, List[HistoricalDataPoint]] = Field(
        ..., description="Historical price data for assets"
    )

class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint."""
    exposures: Dict[str, float] = Field(..., description="Calculated exposures for assets")
    changes: Dict[str, float] = Field(..., description="Day-over-day percentage changes")
    volatility: Dict[str, float] = Field(..., description="Volatility calculations")
    summary: str = Field(..., description="Summary of analysis with alerts")