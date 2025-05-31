# agents/analysis_agent/config.py
# Applying LOG_LEVEL and minor cleanups if any.

import os
from typing import List, Optional, Dict, Any # Dict, Any were not used, but fine to keep
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict # Added SettingsConfigDict
import logging # Added

load_dotenv()

class Settings(BaseSettings):
    VOLATILITY_WINDOW: int = 10 # Days for volatility calculation
    ALERT_THRESHOLD: float = 0.05 # e.g., 5% change/exposure/volatility to be flagged
    
    ANALYSIS_PROVIDER: str = os.getenv("ANALYSIS_PROVIDER", "default")
    FALLBACK_PROVIDERS: List[str] = ["default", "advanced"] # Order of fallback
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8007")) # Ensure type casting for env vars
    
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600")) 
    
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100")) # Requests per minute
    
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() # Ensure uppercase
    
    ENABLE_CORRELATION_ANALYSIS: bool = os.getenv("ENABLE_CORRELATION_ANALYSIS", "True").lower() == "true"
    ENABLE_RISK_METRICS: bool = os.getenv("ENABLE_RISK_METRICS", "True").lower() == "true"

    # model_config was defined as class Config before, Pydantic V2 uses model_config directly
    model_config = SettingsConfigDict(
        env_file=".env", # Added for consistency if .env is used
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Configure logging
log_level_to_set = settings.LOG_LEVEL
if not hasattr(logging, log_level_to_set):
    logging.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in Analysis Agent settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s (ANALYSIS_AGENT) - %(levelname)s - %(message)s"
)
# Re-get logger after basicConfig if this module needs its own logger instance earlier than main.py
# logger = logging.getLogger(__name__) # Not strictly needed here if main configures root