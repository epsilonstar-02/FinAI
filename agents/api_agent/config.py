import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Base URL for the market data API
    BASE_URL: str = os.getenv("BASE_URL", "https://www.alphavantage.co/query")
    # API key for authentication
    API_KEY: str = os.getenv("ALPHAVANTAGE_KEY", "")
    # HTTP timeout (seconds)
    TIMEOUT: int = int(os.getenv("TIMEOUT", 5))


settings = Settings()
