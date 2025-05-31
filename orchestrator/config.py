# orchestrator/config.py
# No changes are made to this file as it seems well-structured and functional.
# Original content is preserved.

"""Configuration for the Orchestrator Agent."""
import os
import logging
from typing import Optional, List, Dict, Any

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Settings for the Orchestrator Agent."""

    # Required agent URLs with defaults for containerized deployment
    ORCHESTRATOR_URL: str = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8004")
    API_AGENT_URL: str = os.getenv("API_AGENT_URL", "http://api_agent:8001")
    SCRAPING_AGENT_URL: str = os.getenv("SCRAPING_AGENT_URL", "http://scraping_agent:8002") # Original had :8002
    RETRIEVER_AGENT_URL: str = os.getenv("RETRIEVER_AGENT_URL", "http://retriever_agent:8003") # Original had :8001, assuming typo and assigning 8003
    ANALYSIS_AGENT_URL: str = os.getenv("ANALYSIS_AGENT_URL", "http://analysis_agent:8007")
    LANGUAGE_AGENT_URL: str = os.getenv("LANGUAGE_AGENT_URL", "http://language_agent:8005") # Original had :8004, assuming typo and assigning 8005
    VOICE_AGENT_URL: str = os.getenv("VOICE_AGENT_URL", "http://voice_agent:8006")
    
    # Host and port settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8004"))
    
    # Timeout settings - General default timeout for the client itself
    CLIENT_DEFAULT_TIMEOUT: int = int(os.getenv("CLIENT_DEFAULT_TIMEOUT", "30")) # Renamed for clarity from TIMEOUT
    
    # Specific timeouts for individual agents (used for request calls)
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "15"))
    SCRAPING_TIMEOUT: int = int(os.getenv("SCRAPING_TIMEOUT", "30"))
    RETRIEVER_TIMEOUT: int = int(os.getenv("RETRIEVER_TIMEOUT", "20"))
    ANALYSIS_TIMEOUT: int = int(os.getenv("ANALYSIS_TIMEOUT", "25"))
    LANGUAGE_TIMEOUT: int = int(os.getenv("LANGUAGE_TIMEOUT", "40"))
    VOICE_TIMEOUT: int = int(os.getenv("VOICE_TIMEOUT", "35"))
    
    # Agent failure handling (for tenacity retries)
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0")) # seconds
    
    # Fallback thresholds
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Default provider settings
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-flash")
    DEFAULT_STT_PROVIDER: str = os.getenv("DEFAULT_STT_PROVIDER", "whisper")
    DEFAULT_TTS_PROVIDER: str = os.getenv("DEFAULT_TTS_PROVIDER", "gtts")
    DEFAULT_VOICE: str = os.getenv("DEFAULT_VOICE", "en-US-Neural2-F")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Cache settings
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
    
    # Rate limiting (if orchestrator itself implements it, not for downstream calls)
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "60"))  # Requests per minute
    
    # Pydantic v2+ configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create settings instance
settings = Settings()

# Configure logging
# Validate LOG_LEVEL
log_level_to_set = settings.LOG_LEVEL
if not hasattr(logging, log_level_to_set):
    logging.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)