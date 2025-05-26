from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Base URL for the market data API
    BASE_URL: str = Field(default="https://www.alphavantage.co/query", env="BASE_URL")
    # API key for authentication
    API_KEY: str = Field(default="", env="ALPHAVANTAGE_KEY")
    # HTTP timeout (seconds)
    TIMEOUT: int = Field(default=5, env="TIMEOUT")

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields

settings = Settings()
