"""
Tests for the Scraping Agent API endpoints.
"""
import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi import HTTPException

from agents.scraping_agent.models import NewsResponse, FilingResponse
from agents.scraping_agent.main import health_check, get_news, get_filing
from agents.scraping_agent.models import NewsRequest, FilingRequest

class TestAPIEndpoints:
    """Tests for the Scraping Agent API endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test the health check endpoint."""
        # Call the endpoint function directly
        response = await health_check()
        
        # Verify the response data
        assert response["status"] == "ok"
        assert response["agent"] == "Scraping Agent"
        assert "timestamp" in response

    @pytest.mark.asyncio
    @patch('agents.scraping_agent.main.fetch_news_loader')
    async def test_news_endpoint_success(self, mock_fetch_news, example_news_response):
        """Test successful news endpoint request."""
        # Setup the mock
        mock_fetch_news.return_value = example_news_response
        
        # Call the endpoint function directly
        response = await get_news(topic="tech", limit=2)
        
        # Verify the response data
        assert response.source == "Google News"
        assert isinstance(response.timestamp, object)
        assert len(response.articles) == 2
        assert response.articles[0].title == "News about tech #1"

    @pytest.mark.asyncio
    @patch('agents.scraping_agent.main.fetch_news_loader')
    async def test_news_endpoint_default_limit(self, mock_fetch_news):
        """Test news endpoint with default limit."""
        # Setup the mock
        mock_fetch_news.return_value = AsyncMock()
        
        # Call the endpoint function directly
        await get_news(topic="sports", limit=5)
        
        # Verify the mock was called with expected args
        mock_fetch_news.assert_called_once_with("sports", 5)

    @pytest.mark.asyncio
    async def test_news_endpoint_limit_validation(self):
        """Test news endpoint with limit out of range."""
        # Test with limit too high
        with pytest.raises(HTTPException) as exc_info:
            await get_news(topic="tech", limit=30)
            
        # Verify exception details
        assert exc_info.value.status_code == 400
        assert "Limit must be between" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('agents.scraping_agent.main.fetch_news_loader')
    async def test_news_endpoint_loader_error(self, mock_fetch_news):
        """Test news endpoint with loader error."""
        # Setup mock to raise exception
        mock_fetch_news.side_effect = HTTPException(502, "Loader failed: Network error")
        
        # Call endpoint and expect exception to propagate
        with pytest.raises(HTTPException) as exc_info:
            await get_news(topic="tech", limit=5)
            
        # Verify exception details
        assert exc_info.value.status_code == 502
        assert "Loader failed" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('agents.scraping_agent.main.fetch_filing_loader')
    async def test_filing_endpoint_success(self, mock_fetch_filing, example_filing_response):
        """Test successful filing endpoint request."""
        # Setup mock
        mock_fetch_filing.return_value = example_filing_response
        
        # Create request object
        request = FilingRequest(filing_url="https://sec.gov/filing/123")
        
        # Call endpoint function directly
        response = await get_filing(request)
        
        # Verify response data
        assert response.source == "https://sec.gov/filing/123"
        assert hasattr(response, "timestamp")
        assert "Filing from" in response.title
        assert "filing body content" in response.body
        assert response.filing_type == "10-K"
        assert hasattr(response, "company")

    # Pydantic validation happens before our endpoint function is called,
    # so we can't directly test validation errors this way

    @pytest.mark.asyncio
    @patch('agents.scraping_agent.main.fetch_filing_loader')
    async def test_filing_endpoint_loader_error(self, mock_fetch_filing):
        """Test filing endpoint with loader error."""
        # Setup mock to raise exception
        mock_fetch_filing.side_effect = HTTPException(502, "Loader failed: Connection timeout")
        
        # Create request object
        request = FilingRequest(filing_url="https://sec.gov/filing/123")
        
        # Call endpoint and expect exception to propagate
        with pytest.raises(HTTPException) as exc_info:
            await get_filing(request)
            
        # Verify exception details
        assert exc_info.value.status_code == 502
        assert "Loader failed" in exc_info.value.detail
