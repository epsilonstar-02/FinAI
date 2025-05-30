"""Models for the Analysis Agent."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator

class HistoricalDataPoint(BaseModel):
    """Model for a single historical data point."""
    date: str = Field(..., description="Date in ISO format (YYYY-MM-DD)")
    close: float = Field(..., description="Closing price")
    open: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="High price")
    low: Optional[float] = Field(None, description="Low price")
    volume: Optional[int] = Field(None, description="Trading volume")

class RiskMetrics(BaseModel):
    """Model for risk metrics of an asset."""
    sharpe_ratio: float = Field(..., description="Sharpe ratio (risk-adjusted return)")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    var_95: float = Field(..., description="Value at Risk (95% confidence)")
    beta: float = Field(..., description="Beta (market sensitivity)")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio (downside risk-adjusted return)")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio (return to max drawdown)")
    cvar_95: Optional[float] = Field(None, description="Conditional Value at Risk (95% confidence)")

class ProviderInfo(BaseModel):
    """Information about the provider used for analysis."""
    name: str = Field(..., description="Provider name")
    version: str = Field("1.0.0", description="Provider version")
    fallback_used: bool = Field(False, description="Whether a fallback provider was used")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")

class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint."""
    prices: Dict[str, float] = Field(..., description="Current prices for assets")
    historical: Dict[str, List[HistoricalDataPoint]] = Field(
        ..., description="Historical price data for assets"
    )
    provider: Optional[str] = Field(None, description="Specific provider to use for analysis")
    include_correlations: Optional[bool] = Field(False, description="Whether to include correlation analysis")
    include_risk_metrics: Optional[bool] = Field(False, description="Whether to include risk metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "prices": {"AAPL": 150.25, "MSFT": 245.80, "GOOGL": 2750.15},
                "historical": {
                    "AAPL": [
                        {"date": "2023-05-01", "close": 150.25},
                        {"date": "2023-04-30", "close": 149.50}
                    ]
                },
                "provider": "advanced",
                "include_correlations": True,
                "include_risk_metrics": True
            }
        }

class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint."""
    exposures: Dict[str, float] = Field(..., description="Calculated exposures for assets")
    changes: Dict[str, float] = Field(..., description="Day-over-day percentage changes")
    volatility: Dict[str, float] = Field(..., description="Volatility calculations")
    correlations: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation matrix between assets")
    risk_metrics: Optional[Dict[str, RiskMetrics]] = Field(None, description="Risk metrics for each asset")
    summary: str = Field(..., description="Summary of analysis with alerts")
    provider_info: ProviderInfo = Field(..., description="Information about the provider used")
    
    class Config:
        schema_extra = {
            "example": {
                "exposures": {"AAPL": 0.15, "MSFT": 0.25, "GOOGL": 0.60},
                "changes": {"AAPL": 0.005, "MSFT": -0.002, "GOOGL": 0.01},
                "volatility": {"AAPL": 0.02, "MSFT": 0.018, "GOOGL": 0.025},
                "correlations": {
                    "AAPL": {"AAPL": 1.0, "MSFT": 0.7, "GOOGL": 0.5},
                    "MSFT": {"AAPL": 0.7, "MSFT": 1.0, "GOOGL": 0.6},
                    "GOOGL": {"AAPL": 0.5, "MSFT": 0.6, "GOOGL": 1.0}
                },
                "risk_metrics": {
                    "AAPL": {
                        "sharpe_ratio": 1.2,
                        "max_drawdown": 0.15,
                        "var_95": -0.025,
                        "beta": 0.9,
                        "sortino_ratio": 1.5,
                        "calmar_ratio": 0.8,
                        "cvar_95": -0.03
                    }
                },
                "summary": "Analysis Summary:\n- High exposure assets: GOOGL\n- Significant price changes: None\n- High volatility assets: GOOGL",
                "provider_info": {
                    "name": "advanced",
                    "version": "1.0.0",
                    "fallback_used": False,
                    "execution_time_ms": 150.5
                }
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response model."""
    status: str = Field("error", description="Status of the response")
    message: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Failed to analyze data",
                "details": "Invalid historical data format"
            }
        }