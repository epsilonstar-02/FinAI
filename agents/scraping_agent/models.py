"""
Data models for the Scraping Agent microservice.
Defines Pydantic models for requests and responses.
"""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field

class NewsRequest(BaseModel):
    """Request model for news scraping."""
    topic: str = Field(..., description="News topic to search for")
    limit: int = Field(5, description="Maximum number of articles to return", ge=1, le=20)

class NewsArticle(BaseModel):
    """Model representing a news article."""
    title: str
    body: str
    url: HttpUrl
    source: str
    published_date: Optional[datetime] = None

class NewsResponse(BaseModel):
    """Response model for news scraping."""
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    articles: List[NewsArticle]
    
class FilingRequest(BaseModel):
    """Request model for SEC filing scraping."""
    filing_url: HttpUrl = Field(..., description="URL of the SEC filing to scrape")
    
class FilingResponse(BaseModel):
    """Response model for SEC filing scraping."""
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    title: str
    body: str
    filing_type: Optional[str] = None
    filing_date: Optional[datetime] = None
    company: Optional[str] = None
