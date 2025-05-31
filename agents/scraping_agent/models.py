# agents/scraping_agent/models.py
# No significant changes needed. Models are well-defined.
# Added default_factory for timestamps.
# Ensured constr pattern for symbol is consistent.

from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, HttpUrl, Field, constr

class NewsRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="News topic to search for")
    limit: int = Field(5, ge=1, le=25, description="Max articles (1-25)") # Max limit adjusted
    source: Optional[str] = Field(None, description="Preferred news source (e.g., 'google', 'yahoo'). If None, defaults to Google News.")

class CompanyNewsRequest(BaseModel): # Not directly used if path param is used, but good for reference
    symbol: constr(min_length=1, max_length=10, pattern=r'^[A-Z0-9.\-]+$')
    limit: int = Field(5, ge=1, le=25)

class MarketNewsRequest(BaseModel):
    limit: int = Field(5, ge=1, le=30) # Market news can have slightly higher limit
    category: Optional[str] = Field(None, description="E.g., 'stocks', 'crypto', 'ipo', 'mergers', 'economy'")

class FilingRequest(BaseModel): # For fetching a single filing by URL
    filing_url: HttpUrl

# Response Models
class NewsArticle(BaseModel):
    title: str
    body: str 
    url: HttpUrl
    source: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    summary: Optional[str] = None # Often, body itself is the summary if full text not extracted.

class NewsResponse(BaseModel):
    source: str # Name of the primary news source used
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    articles: List[NewsArticle]
    query: Optional[str] = None # The original query topic or symbol

class CompanyNewsResponse(NewsResponse):
    symbol: str
    company_name: Optional[str] = None

class MarketNewsResponse(NewsResponse):
    category: Optional[str] = None

class FilingResponse(BaseModel):
    source: str # URL of the filing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    title: str
    body: str # Extracted text content of the filing
    filing_type: Optional[str] = None # E.g., "10-K", "8-K"
    filing_date: Optional[datetime] = None # Date the filing refers to or was filed
    company: Optional[str] = None # Company name from filing
    symbol: Optional[str] = None # Symbol, if derivable or passed in context

class CompanyFilingsResponse(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    filings: List[FilingResponse] # List of individual filing data

class CompanyProfileResponse(BaseModel):
    symbol: str
    name: Optional[str] = None # Optional if not found
    description: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[HttpUrl] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    price: Optional[float] = None # Current or recent price
    employees: Optional[int] = None
    country: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EarningsResponse(BaseModel):
    symbol: str
    company_name: Optional[str] = None # Optional if not found
    earnings_date: Optional[datetime] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    quarter: Optional[int] = None
    year: Optional[int] = None
    surprise_percent: Optional[float] = None
    transcript_url: Optional[HttpUrl] = None # URL to transcript, if found
    transcript_text: Optional[str] = None # Extracted transcript text, if processed
    timestamp: datetime = Field(default_factory=datetime.utcnow)