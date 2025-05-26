"""
Scraping Agent microservice for FinAI.
Provides functionality for scraping news and SEC filings.
"""
from .config import settings
from .models import NewsRequest, NewsResponse, FilingRequest, FilingResponse
from .client_loader import fetch_news_loader, fetch_filing_loader

__version__ = "1.0.0"
