"""Configuration for the Analysis Agent."""
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import logging

load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for the Analysis Agent."""
    # Analysis settings
    VOLATILITY_WINDOW: int = 10
    ALERT_THRESHOLD: float = 0.05
    
    # Provider settings
    ANALYSIS_PROVIDER: str = os.getenv("ANALYSIS_PROVIDER", "default")
    FALLBACK_PROVIDERS: List[str] = ["default", "advanced"]
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8007"))
    
    # Caching settings
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100"))  # Requests per minute
    
    # Timeouts
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))  # Seconds
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Feature flags
    ENABLE_CORRELATION_ANALYSIS: bool = os.getenv("ENABLE_CORRELATION_ANALYSIS", "True").lower() == "true"
    ENABLE_RISK_METRICS: bool = os.getenv("ENABLE_RISK_METRICS", "True").lower() == "true"

    class Config:
        extra = "ignore"

# Initialize settings
settings = Settings()

# Configure logging
logging_level = getattr(logging, settings.LOG_LEVEL)
logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')