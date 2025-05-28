"""
Tests for the enhanced Scraping Agent API endpoints.
Validates the functionality of the FastAPI endpoints with mocked responses.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Import the FastAPI app
from agents.scraping_agent.main import app
from agents.scraping_agent.models import (
    NewsArticle, NewsResponse, CompanyNewsResponse, MarketNewsResponse,
    FilingResponse, CompanyFilingsResponse, CompanyProfileResponse,
    EarningsResponse
)
from datetime import datetime

# Create test client
client = TestClient(app)

# Mock data for responses
mock_news_article = NewsArticle(
    title="Test Article",
    body="This is test content for the article",
    url="https://example.com/news/1",
    source="Test Source",
    published_date=datetime.now()
)

mock_filing_response = FilingResponse(
    source="https://sec.gov/example",
    timestamp=datetime.now(),
    title="SEC Filing",
    body="FORM 10-K\nTest filing content",
    filing_type="10-K",
    filing_date=datetime.now(),
    company="Test Corp",
    symbol="TEST"
)

mock_company_profile = CompanyProfileResponse(
    symbol="TEST",
    name="Test Company Inc.",
    description="Test company description",
    sector="Technology",
    industry="Software",
    website="https://testcompany.com",
    market_cap=1000000000,
    pe_ratio=20.5,
    price=150.75,
    employees=5000,
    country="United States",
    timestamp=datetime.now()
)

mock_earnings_response = EarningsResponse(
    symbol="TEST",
    company_name="Test Company Inc.",
    earnings_date=datetime.now(),
    eps_estimate=2.75,
    eps_actual=2.5,
    revenue_estimate=1050000000,
    revenue_actual=1000000000,
    quarter=1,
    year=2025,
    surprise_percent=-9.09,
    timestamp=datetime.now()
)

# Tests

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["agent"] == "Enhanced Scraping Agent"
    assert "version" in data
    assert "features" in data

@patch("agents.scraping_agent.document_loaders.news_loader.fetch_google_news")
def test_get_news_endpoint(mock_fetch_news):
    """Test the /news endpoint"""
    # Configure mock
    mock_fetch_news.return_value = AsyncMock(return_value=[mock_news_article])()
    
    # Test the endpoint
    response = client.get("/news?topic=test&limit=1")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "Google News"
    assert "articles" in data
    assert len(data["articles"]) >= 1
    assert data["articles"][0]["title"] == "Test Article"

@patch("agents.scraping_agent.document_loaders.news_loader.fetch_yahoo_finance_news")
@patch("agents.scraping_agent.document_loaders.company_profile_loader.fetch_company_profile")
def test_get_company_news_endpoint(mock_profile, mock_fetch_news):
    """Test the /company/news/{symbol} endpoint"""
    # Configure mocks
    mock_fetch_news.return_value = AsyncMock(return_value=[mock_news_article])()
    mock_profile.return_value = AsyncMock(return_value=mock_company_profile)()
    
    # Test the endpoint
    response = client.get("/company/news/TEST?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TEST"
    assert "articles" in data
    assert len(data["articles"]) >= 1

@patch("agents.scraping_agent.document_loaders.news_loader.fetch_market_news")
def test_get_market_news_endpoint(mock_fetch_news):
    """Test the /market/news endpoint"""
    # Configure mock
    mock_fetch_news.return_value = AsyncMock(return_value=[mock_news_article])()
    
    # Test the endpoint
    response = client.get("/market/news?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert "articles" in data
    assert len(data["articles"]) >= 1

@patch("agents.scraping_agent.document_loaders.sec_filing_loader._process_filing")
def test_get_filing_endpoint(mock_process_filing):
    """Test the /filing endpoint"""
    # Configure mock
    mock_process_filing.return_value = AsyncMock(return_value=mock_filing_response)()
    
    # Test the endpoint
    response = client.post("/filing", json={"filing_url": "https://sec.gov/example"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "https://sec.gov/example"
    assert data["filing_type"] == "10-K"
    assert "Test filing content" in data["body"]

@patch("agents.scraping_agent.document_loaders.sec_filing_loader.fetch_company_filings")
@patch("agents.scraping_agent.document_loaders.company_profile_loader.fetch_company_profile")
def test_get_company_filings_endpoint(mock_profile, mock_fetch_filings):
    """Test the /company/filings/{symbol} endpoint"""
    # Configure mocks
    mock_fetch_filings.return_value = AsyncMock(return_value=[mock_filing_response])()
    mock_profile.return_value = AsyncMock(return_value=mock_company_profile)()
    
    # Test the endpoint
    response = client.get("/company/filings/TEST?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TEST"
    assert "filings" in data
    assert len(data["filings"]) >= 1
    assert data["filings"][0]["filing_type"] == "10-K"

@patch("agents.scraping_agent.document_loaders.company_profile_loader.fetch_company_profile")
def test_get_company_profile_endpoint(mock_fetch_profile):
    """Test the /company/profile/{symbol} endpoint"""
    # Configure mock
    mock_fetch_profile.return_value = AsyncMock(return_value=mock_company_profile)()
    
    # Test the endpoint
    response = client.get("/company/profile/TEST")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TEST"
    assert data["name"] == "Test Company Inc."
    assert data["sector"] == "Technology"
    assert data["industry"] == "Software"
    assert data["market_cap"] == 1000000000
    assert data["price"] == 150.75

@patch("agents.scraping_agent.document_loaders.earnings_loader.fetch_latest_earnings")
def test_get_company_earnings_endpoint(mock_fetch_earnings):
    """Test the /company/earnings/{symbol} endpoint"""
    # Configure mock
    mock_fetch_earnings.return_value = AsyncMock(return_value=mock_earnings_response)()
    
    # Test the endpoint
    response = client.get("/company/earnings/TEST")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TEST"
    assert data["company_name"] == "Test Company Inc."
    assert data["eps_estimate"] == 2.75
    assert data["eps_actual"] == 2.5
    assert data["revenue_estimate"] == 1050000000
    assert data["revenue_actual"] == 1000000000
    assert data["quarter"] == 1
    assert data["year"] == 2025

def test_error_handling():
    """Test error handling for invalid requests"""
    # Test invalid symbol
    response = client.get("/company/profile/INVALID_SYMBOL_TOO_LONG")
    assert response.status_code == 422  # Validation error
    
    # Test invalid limit
    response = client.get("/news?topic=test&limit=100")  # Limit too high
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
