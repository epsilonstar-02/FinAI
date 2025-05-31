# agents/scraping_agent/config.py
# No significant changes needed. Ensure LOG_LEVEL is applied.

from pydantic_settings import BaseSettings, SettingsConfigDict # Added SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv
import logging # Added

load_dotenv()

class Settings(BaseSettings):
    USER_AGENT: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 FinAI/1.0" # More specific user agent
    TIMEOUT: int = 15 # Increased default timeout slightly
    MAX_RETRIES: int = 2 # Default retries for scraping operations (e.g., in extract_with_multiple_methods)
    
    ALPHAVANTAGE_API_KEY: Optional[str] = None # Not directly used by this agent's current loaders
    SEC_API_KEY: Optional[str] = None # Used as User-Agent for secedgar, should be an email.
                                      # Format: "Sample Company Name AdminContact@<sample company domain>.com"
    
    HOST: str = "0.0.0.0"
    PORT: int = 8002 # Original port was 8002, Orchestrator uses 8001. This should be 8002 based on Orchestrator config.
    LOG_LEVEL: str = "INFO" # Default log level
    
    model_config = SettingsConfigDict( # Pydantic V2 style
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

# Configure logging
log_level_to_set = settings.LOG_LEVEL.upper()
if not hasattr(logging, log_level_to_set):
    logging.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in Scraping Agent settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s (SCRAPING_AGENT) - %(levelname)s - %(message)s"
)