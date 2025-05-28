"""Configuration for the Orchestrator Agent."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the Orchestrator Agent."""

    # Required agent URLs with defaults for local development
    API_AGENT_URL: str = "http://localhost:8000"
    SCRAPING_AGENT_URL: str = "http://localhost:8001"
    RETRIEVER_AGENT_URL: str = "http://localhost:8002"
    ANALYSIS_AGENT_URL: str = "http://localhost:8003"
    LANGUAGE_AGENT_URL: str = "http://localhost:8004"
    VOICE_AGENT_URL: str = "http://localhost:8005"
    
    # Timeout settings
    TIMEOUT: int = 5
    
    # Pydantic v2+ configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields from env vars meant for other services
    )


# Create settings instance
settings = Settings()
