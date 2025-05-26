"""
Tests for the Scraping Agent document loaders.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from httpx import Response

from agents.scraping_agent.client_loader import fetch_news_loader, fetch_filing_loader
from agents.scraping_agent.models import NewsResponse, FilingResponse

@pytest.mark.asyncio
class TestClientLoader:
    """Tests for the client_loader module."""
    
    @patch('agents.scraping_agent.client_loader.UnstructuredURLLoader')
    @patch('agents.scraping_agent.client_loader.httpx.AsyncClient')
    async def test_fetch_news_loader_success(self, mock_client, mock_loader_class, 
                                           mock_news_documents):
        """Test successful news loading from RSS."""
        # Mock HTTP client for RSS feed
        mock_client_instance = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.text = """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Google News - Tech</title>
                <item>
                    <title>Tech News Article 1</title>
                    <link>https://example.com/article/1</link>
                    <pubDate>Mon, 26 May 2025 12:00:00 GMT</pubDate>
                    <description>Article 1 description</description>
                </item>
                <item>
                    <title>Tech News Article 2</title>
                    <link>https://example.com/article/2</link>
                    <pubDate>Mon, 26 May 2025 11:00:00 GMT</pubDate>
                    <description>Article 2 description</description>
                </item>
            </channel>
        </rss>
        """
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        # Mock UnstructuredURLLoader for article content
        mock_loader = Mock()
        mock_loader.load.return_value = mock_news_documents
        mock_loader_class.return_value = mock_loader
        
        # Call the function
        result = await fetch_news_loader("tech", 2)
        
        # Verify results
        assert isinstance(result, NewsResponse)
        assert result.source == "Google News"
        assert isinstance(result.timestamp, datetime)
        assert len(result.articles) == 2
        
        # Verify RSS feed was requested
        mock_client_instance.get.assert_called_once()
        assert "news.google.com" in mock_client_instance.get.call_args[0][0]
        assert "tech" in mock_client_instance.get.call_args[0][0]

    @patch('agents.scraping_agent.client_loader.UnstructuredURLLoader')
    @patch('agents.scraping_agent.client_loader.httpx.AsyncClient')
    async def test_fetch_news_loader_fallback(self, mock_client, mock_loader_class,
                                            mock_news_documents):
        """Test fallback to UnstructuredURLLoader when RSS parsing fails."""
        # Mock HTTP client for RSS feed with invalid XML
        mock_client_instance = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.text = "Not XML content"
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        # Mock UnstructuredURLLoader
        mock_loader = Mock()
        mock_loader.load.return_value = mock_news_documents
        mock_loader_class.return_value = mock_loader
        
        # Call the function
        result = await fetch_news_loader("tech", 1)
        
        # Verify results
        assert isinstance(result, NewsResponse)
        assert result.source == "Google News"
        assert len(result.articles) == 1
        
        # Verify UnstructuredURLLoader was used as fallback
        mock_loader_class.assert_called_once()
        assert "news.google.com" in mock_loader_class.call_args[1]["urls"][0]

    @patch('agents.scraping_agent.client_loader.httpx.AsyncClient')
    async def test_fetch_news_loader_http_error(self, mock_client):
        """Test handling of HTTP errors."""
        # Mock HTTP client to raise an exception
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = Exception("Connection error")
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        # Call the function and expect an exception
        with pytest.raises(HTTPException) as exc_info:
            await fetch_news_loader("tech", 5)
        
        # Verify exception details
        assert exc_info.value.status_code == 502
        assert "News loader failed" in exc_info.value.detail
        assert "Connection error" in exc_info.value.detail

    @patch('agents.scraping_agent.client_loader.UnstructuredURLLoader')
    async def test_fetch_filing_loader_success(self, mock_loader_class,
                                             mock_filing_document):
        """Test successful SEC filing loading."""
        # Mock UnstructuredURLLoader
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_filing_document]
        mock_loader_class.return_value = mock_loader
        
        # Call the function
        url = "https://sec.gov/filing/123"
        result = await fetch_filing_loader(url)
        
        # Verify results
        assert isinstance(result, FilingResponse)
        assert result.source == url
        assert isinstance(result.timestamp, datetime)
        assert result.title == f"Filing from {url}"
        assert result.body == "SEC Filing Title\n\nThis is the filing body content."
        
        # Verify loader was called with correct URL
        mock_loader_class.assert_called_once()
        assert url in mock_loader_class.call_args[1]["urls"][0]

    @patch('agents.scraping_agent.client_loader.UnstructuredURLLoader')
    async def test_fetch_filing_loader_no_docs(self, mock_loader_class):
        """Test handling of empty document list."""
        # Mock UnstructuredURLLoader to return empty list
        mock_loader = Mock()
        mock_loader.load.return_value = []
        mock_loader_class.return_value = mock_loader
        
        # Call the function and expect an exception
        with pytest.raises(HTTPException) as exc_info:
            await fetch_filing_loader("https://sec.gov/filing/123")
        
        # Verify exception details
        assert exc_info.value.status_code == 502
        assert "Filing loader failed" in exc_info.value.detail
        assert "No content loaded" in exc_info.value.detail

    @patch('agents.scraping_agent.client_loader.UnstructuredURLLoader')
    async def test_fetch_filing_loader_failure(self, mock_loader_class):
        """Test handling of loader failures."""
        # Mock UnstructuredURLLoader to raise an exception
        mock_loader_class.side_effect = Exception("Connection timeout")
        
        # Call the function and expect an exception
        with pytest.raises(HTTPException) as exc_info:
            await fetch_filing_loader("https://sec.gov/filing/123")
        
        # Verify exception details
        assert exc_info.value.status_code == 502
        assert "Filing loader failed" in exc_info.value.detail
        assert "Connection timeout" in exc_info.value.detail
