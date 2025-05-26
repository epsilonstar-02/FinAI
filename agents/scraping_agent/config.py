"""
Configuration for the Scraping Agent microservice.
Uses pydantic-settings for type-safe environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variables support."""
    # Scraping settings
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    
    # API Keys
    ALPHAVANTAGE_API_KEY: Optional[str] = None
    SEC_API_KEY: Optional[str] = None
    
    # Service settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "info"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()
