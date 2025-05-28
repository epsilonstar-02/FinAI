"""Configuration for the Analysis Agent."""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for the Analysis Agent."""
    VOLATILITY_WINDOW: int = 10
    ALERT_THRESHOLD: float = 0.05

    class Config:
        extra = "ignore"

settings = Settings()