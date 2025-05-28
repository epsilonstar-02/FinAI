from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Dict, List
from enum import Enum

class DataProvider(str, Enum):
    """Supported financial data providers"""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    FMP = "financial_modeling_prep"

class Settings(BaseSettings):
    # Alpha Vantage settings
    ALPHA_VANTAGE_URL: str = Field(default="https://www.alphavantage.co/query", env="ALPHA_VANTAGE_URL")
    ALPHA_VANTAGE_KEY: str = Field(default="", env="ALPHA_VANTAGE_KEY")
    
    # Financial Modeling Prep settings (free tier)
    FMP_URL: str = Field(default="https://financialmodelingprep.com/api/v3", env="FMP_URL")
    FMP_KEY: str = Field(default="", env="FMP_KEY")
    
    # HTTP timeout (seconds)
    TIMEOUT: int = Field(default=5, env="TIMEOUT")
    
    # Provider preferences (ordered list of providers to try)
    PROVIDER_PRIORITY: List[str] = Field(
        default=[DataProvider.YAHOO_FINANCE, DataProvider.ALPHA_VANTAGE, DataProvider.FMP],
        env="PROVIDER_PRIORITY"
    )
    
    # Enable fallback to alternative providers if primary fails
    ENABLE_FALLBACK: bool = Field(default=True, env="ENABLE_FALLBACK")
    
    # Maximum retry attempts per provider
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    
    # Retry backoff factor (seconds)
    RETRY_BACKOFF: float = Field(default=0.5, env="RETRY_BACKOFF")

    # Pydantic v2+ configuration
    model_config = {
        "env_file": ".env",
        "extra": "allow"  # Allow extra fields from env vars meant for other services
    }
    
    # For backward compatibility
    @property
    def BASE_URL(self) -> str:
        return self.ALPHA_VANTAGE_URL
        
    @property
    def API_KEY(self) -> str:
        return self.ALPHA_VANTAGE_KEY

settings = Settings()
