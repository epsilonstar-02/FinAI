from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Configuration settings for the Language Agent.
    
    Attributes:
        GEMINI_API_KEY: API key for Google Generative AI
        GEMINI_MODEL: Model name for Gemini (default: "gemini-flash")
        TIMEOUT: Request timeout in seconds (default: 10)
    """
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-flash"
    TIMEOUT: int = 10
    
    class Config:
        extra = "ignore"


# Create settings instance
settings = Settings()
