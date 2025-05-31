# agents/api_agent/config.py
# No significant changes needed, mostly ensuring LOG_LEVEL is applied.
# Original content is preserved with minor adjustments for log level.

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Union
from enum import Enum
import os
import logging # Added for log level validation

class DataProvider(str, Enum):
    """Supported financial data providers. Values are used as keys/identifiers."""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    FMP = "financial_modeling_prep"

class Settings(BaseSettings):
    # Alpha Vantage settings
    ALPHA_VANTAGE_URL: str = Field(default="https://www.alphavantage.co/query", description="Alpha Vantage API base URL.")
    ALPHA_VANTAGE_KEY: str = Field(default="", description="API key for Alpha Vantage.")
    
    # Financial Modeling Prep settings (free tier)
    FMP_URL: str = Field(default="https://financialmodelingprep.com/api/v3", description="Financial Modeling Prep API base URL.")
    FMP_KEY: str = Field(default="", description="API key for Financial Modeling Prep.")
    
    # Common HTTP client settings
    TIMEOUT: int = Field(default=10, ge=1, le=60, description="HTTP client timeout in seconds.")
    
    # Provider strategy settings
    PROVIDER_PRIORITY: List[DataProvider] = Field(
        default=[
            DataProvider.YAHOO_FINANCE, 
            DataProvider.ALPHA_VANTAGE, 
            DataProvider.FMP
        ],
        description="Ordered list of data providers to try by default."
    )
    
    ENABLE_FALLBACK: bool = Field(default=True, description="Enable fallback to other providers if a preferred/primary one fails.")
    
    # Retry mechanism settings
    MAX_RETRIES: int = Field(default=2, ge=0, le=5, description="Maximum retry attempts for fetching data from a provider sequence.")
    RETRY_BACKOFF: float = Field(default=1.0, ge=0.1, le=5.0, description="Exponential backoff factor for retries in seconds.")

    # Logging level
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).")

    # Pydantic V2 configuration
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding='utf-8',
        extra="ignore"
    )
    
    @field_validator('PROVIDER_PRIORITY', mode='before')
    @classmethod
    def _parse_provider_priority(cls, value: Union[str, List[str]]) -> List[DataProvider]:
        """Allow PROVIDER_PRIORITY from ENV as comma-separated string."""
        if isinstance(value, str):
            value = [item.strip() for item in value.split(',') if item.strip()] # Ensure no empty strings
        
        validated_providers = []
        for item_str in value:
            try:
                provider_enum = DataProvider(item_str)
                validated_providers.append(provider_enum)
            except ValueError:
                # Log a warning instead of raising ValueError during settings load
                # This allows the app to start even if an env var is misconfigured,
                # and the provider will simply be unavailable.
                # Or, keep raising ValueError for stricter startup. For now, warning.
                _temp_logger = logging.getLogger(__name__) # Temp logger for settings validation
                _temp_logger.warning(
                    f"Invalid provider string '{item_str}' in PROVIDER_PRIORITY. "
                    f"Allowed values are: {[dp.value for dp in DataProvider]}. Ignoring this invalid provider."
                )
        return validated_providers

    # Backward compatibility properties (DEPRECATED)
    @property
    def BASE_URL(self) -> str:
        """DEPRECATED: Use specific provider URLs. Returns Alpha Vantage URL for compatibility."""
        # logger.warning("Accessing deprecated settings.BASE_URL. Use specific provider URLs.")
        return self.ALPHA_VANTAGE_URL
        
    @property
    def API_KEY(self) -> str:
        """DEPRECATED: Use specific provider keys. Returns Alpha Vantage key for compatibility."""
        # logger.warning("Accessing deprecated settings.API_KEY. Use specific provider keys.")
        return self.ALPHA_VANTAGE_KEY

settings = Settings()

# Configure logging (ensure this doesn't conflict with orchestrator's logging setup if run in same process)
# This basicConfig should ideally be called once at the highest level of the application.
# For standalone agent testing, it's fine here.
log_level_to_set = settings.LOG_LEVEL.upper()
if not hasattr(logging, log_level_to_set):
    logging.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in API Agent settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s (API_AGENT) - %(levelname)s - %(message)s"
)
# Re-get logger after basicConfig
logger = logging.getLogger(__name__)