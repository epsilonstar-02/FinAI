"""
Pytest fixtures for testing the Scraping Agent.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock
from fastapi.testclient import TestClient

from agents.scraping_agent.models import NewsResponse, NewsArticle, FilingResponse
from agents.scraping_agent.main import app

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    # TestClient initialization with compatibility for different FastAPI/httpx versions
    from fastapi.testclient import TestClient as _TestClient
    
    # Custom TestClient to work around compatibility issues
    class CompatibleTestClient(_TestClient):
        def __init__(self, app):
            super().__init__(app=app)
    
    return CompatibleTestClient(app)

@pytest.fixture
def mock_news_documents():
    """Create mock news documents for testing."""
    docs = []
    for i in range(3):
        doc = Mock()
        doc.page_content = f"This is news article {i+1} content about the topic."
        doc.metadata = {"source": f"https://example.com/news/{i+1}"}
        docs.append(doc)
    return docs

@pytest.fixture
def mock_filing_document():
    """Create a mock filing document for testing."""
    doc = Mock()
    doc.page_content = "SEC Filing Title\n\nThis is the filing body content."
    doc.metadata = {"source": "https://sec.gov/filing/123"}
    return doc

@pytest.fixture
def example_news_response():
    """Create an example news response for testing."""
    articles = [
        NewsArticle(
            title="News about tech #1",
            body="This is news article 1 content about the topic.",
            url="https://example.com/article/1",
            source="Google News"
        ),
        NewsArticle(
            title="News about tech #2", 
            body="This is news article 2 content about the topic.",
            url="https://example.com/article/2",
            source="Google News"
        )
    ]
    return NewsResponse(
        source="Google News",
        timestamp=datetime.now(),
        articles=articles
    )

@pytest.fixture
def example_filing_response():
    """Create an example filing response for testing."""
    return FilingResponse(
        source="https://sec.gov/filing/123",
        timestamp=datetime.now(),
        title="Filing from https://sec.gov/filing/123",
        body="SEC Filing Title\n\nThis is the filing body content.",
        filing_type="10-K",
        filing_date=datetime.now(),
        company="Example Corp"
    )
