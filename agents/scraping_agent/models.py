"""
Enhanced data models for the Scraping Agent microservice.
Defines Pydantic models for requests and responses for various financial data sources.
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, HttpUrl, Field, constr

# Request Models

class NewsRequest(BaseModel):
    """Request model for general news scraping."""
    topic: str = Field(..., description="News topic to search for")
    limit: int = Field(5, description="Maximum number of articles to return", ge=1, le=20)
    source: Optional[str] = Field(None, description="Preferred news source (e.g., 'google', 'yahoo')")

class CompanyNewsRequest(BaseModel):
    """Request model for company-specific news."""
    symbol: constr(min_length=1, max_length=10, pattern=r'^[A-Z.\-]+$') = Field(
        ..., description="Stock ticker symbol"
    )
    limit: int = Field(5, description="Maximum number of articles to return", ge=1, le=20)

class MarketNewsRequest(BaseModel):
    """Request model for general market news."""
    limit: int = Field(5, description="Maximum number of articles to return", ge=1, le=20)
    category: Optional[str] = Field(None, description="News category (e.g., 'stocks', 'crypto', 'economy')")

class FilingRequest(BaseModel):
    """Request model for SEC filing scraping by URL."""
    filing_url: HttpUrl = Field(..., description="URL of the SEC filing to scrape")

class CompanyFilingsRequest(BaseModel):
    """Request model for company SEC filings."""
    symbol: constr(min_length=1, max_length=10, pattern=r'^[A-Z.\-]+$') = Field(
        ..., description="Stock ticker symbol"
    )
    form_type: Optional[str] = Field(None, description="SEC form type (e.g., '10-K', '10-Q', '8-K')")
    limit: int = Field(5, description="Maximum number of filings to return", ge=1, le=10)

class CompanyProfileRequest(BaseModel):
    """Request model for company profile data."""
    symbol: constr(min_length=1, max_length=10, pattern=r'^[A-Z.\-]+$') = Field(
        ..., description="Stock ticker symbol"
    )

class EarningsRequest(BaseModel):
    """Request model for company earnings data."""
    symbol: constr(min_length=1, max_length=10, pattern=r'^[A-Z.\-]+$') = Field(
        ..., description="Stock ticker symbol"
    )

# Response Models

class NewsArticle(BaseModel):
    """Model representing a news article."""
    title: str
    body: str
    url: HttpUrl
    source: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    summary: Optional[str] = None

class NewsResponse(BaseModel):
    """Response model for general news scraping."""
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    articles: List[NewsArticle]
    query: Optional[str] = None

class CompanyNewsResponse(NewsResponse):
    """Response model for company-specific news."""
    symbol: str
    company_name: Optional[str] = None

class MarketNewsResponse(NewsResponse):
    """Response model for market news."""
    category: Optional[str] = None

class FilingResponse(BaseModel):
    """Response model for SEC filing scraping."""
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    title: str
    body: str
    filing_type: Optional[str] = None
    filing_date: Optional[datetime] = None
    company: Optional[str] = None
    symbol: Optional[str] = None

class CompanyFilingsResponse(BaseModel):
    """Response model for company SEC filings."""
    symbol: str
    company_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    filings: List[FilingResponse]

class FinancialMetrics(BaseModel):
    """Model for financial metrics."""
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    return_on_equity: Optional[float] = None

class FinancialReportResponse(BaseModel):
    """Response model for financial reports."""
    symbol: str
    company_name: Optional[str] = None
    report_type: str  # e.g., 'annual', 'quarterly'
    report_date: datetime
    period_end_date: date
    metrics: FinancialMetrics
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)

class CompanyProfileResponse(BaseModel):
    """Response model for company profile."""
    symbol: str
    name: str
    description: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[HttpUrl] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    price: Optional[float] = None
    employees: Optional[int] = None
    country: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class EarningsResponse(BaseModel):
    """Response model for earnings report."""
    symbol: str
    company_name: str
    earnings_date: Optional[datetime] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    quarter: Optional[int] = None
    year: Optional[int] = None
    surprise_percent: Optional[float] = None
    transcript: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
