"""
Tests for the enhanced document loaders in the Scraping Agent.
Validates the functionality of various scraping methods with mocked responses.
"""
import asyncio
import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

# Import the document loaders from the agent
from agents.scraping_agent.document_loaders import (
    DocumentLoader, NewsLoader, SECFilingLoader, 
    CompanyProfileLoader, EarningsLoader,
    ScrapingError, ContentExtractionError
)
from agents.scraping_agent.models import NewsArticle, FilingResponse

# Mock data for testing
MOCK_HTML = """
<html>
<head><title>Test Article</title></head>
<body>
<h1>Test Article Heading</h1>
<p>This is a test article content for scraping tests.</p>
<p>It contains multiple paragraphs with relevant information.</p>
</body>
</html>
"""

MOCK_RSS = """
<rss version="2.0">
<channel>
<title>Test News Feed</title>
<link>https://example.com/news</link>
<description>Test news feed for unit tests</description>
<item>
<title>Test Article 1</title>
<link>https://example.com/news/1</link>
<pubDate>Wed, 21 Apr 2025 15:30:00 GMT</pubDate>
<description>Test article 1 description</description>
</item>
<item>
<title>Test Article 2</title>
<link>https://example.com/news/2</link>
<pubDate>Wed, 21 Apr 2025 14:30:00 GMT</pubDate>
<description>Test article 2 description</description>
</item>
</channel>
</rss>
"""

# Fixtures

@pytest.fixture
def mock_httpx_client():
    """Fixture for mocking httpx client responses"""
    with patch('httpx.AsyncClient') as mock_client:
        # Configure the mock client for different URLs
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_HTML
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        
        # Configure async context manager behavior
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.get.return_value = mock_response
        
        mock_client.return_value = mock_instance
        yield mock_client

@pytest.fixture
def mock_news_rss():
    """Fixture for mocking RSS feed responses"""
    with patch('httpx.AsyncClient') as mock_client:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RSS
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        
        # Configure async context manager behavior
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.get.return_value = mock_response
        
        mock_client.return_value = mock_instance
        yield mock_client

@pytest.fixture
def mock_trafilatura():
    """Fixture for mocking trafilatura extraction"""
    with patch('trafilatura.extract') as mock_extract:
        mock_extract.return_value = '{"title": "Test Article", "text": "This is test content", "author": "Test Author", "date": "2025-04-21"}'
        yield mock_extract

@pytest.fixture
def mock_yfinance():
    """Fixture for mocking yfinance data"""
    with patch('yfinance.Ticker') as mock_ticker:
        # Create mock ticker instance
        ticker_instance = MagicMock()
        
        # Mock the news property
        ticker_instance.news = [
            {
                'title': 'Test Company News',
                'link': 'https://finance.example.com/news/1',
                'publisher': 'Test Publisher',
                'providerPublishTime': int(datetime.now().timestamp())
            },
            {
                'title': 'Another Test News',
                'link': 'https://finance.example.com/news/2',
                'publisher': 'Another Publisher',
                'providerPublishTime': int(datetime.now().timestamp())
            }
        ]
        
        # Mock the info property
        ticker_instance.info = {
            'longName': 'Test Company Inc.',
            'shortName': 'TEST',
            'sector': 'Technology',
            'industry': 'Software',
            'website': 'https://testcompany.com',
            'marketCap': 1000000000,
            'trailingPE': 20.5,
            'currentPrice': 150.75,
            'fullTimeEmployees': 5000,
            'country': 'United States'
        }
        
        # Mock the earnings property
        ticker_instance.earnings = {
            'quarterly': MagicMock()
        }
        ticker_instance.earnings['quarterly'].empty = False
        ticker_instance.earnings['quarterly'].iloc = [MagicMock()]
        ticker_instance.earnings['quarterly'].iloc[-1].get.side_effect = lambda key, default: 2.5 if key == 'Earnings' else 1000000000
        
        # Mock the calendar property
        ticker_instance.calendar = {
            'Earnings Date': [datetime.now()],
            'EPS Estimate': [2.75],
            'Revenue Estimate': [1050000000]
        }
        
        mock_ticker.return_value = ticker_instance
        yield mock_ticker

# Tests for the DocumentLoader base class

@pytest.mark.asyncio
async def test_document_loader_get_content(mock_httpx_client):
    """Test the DocumentLoader.get_content method"""
    loader = DocumentLoader()
    content = await loader.get_content("https://example.com")
    
    # Verify the response
    assert content == MOCK_HTML
    mock_httpx_client.assert_called_once()

@pytest.mark.asyncio
async def test_document_loader_extract_with_multiple_methods(mock_httpx_client, mock_trafilatura):
    """Test the DocumentLoader.extract_with_multiple_methods method"""
    loader = DocumentLoader()
    result = await loader.extract_with_multiple_methods("https://example.com")
    
    # Verify the extraction
    assert "title" in result
    assert "text" in result
    assert "source" in result
    mock_trafilatura.assert_called_once()

@pytest.mark.asyncio
async def test_document_loader_extraction_fallback(mock_httpx_client):
    """Test the extraction fallback mechanisms when primary method fails"""
    loader = DocumentLoader()
    
    # Mock trafilatura to fail
    with patch('trafilatura.extract', return_value=None):
        # Mock newspaper to fail
        with patch('newspaper.Article', side_effect=Exception("Newspaper error")):
            # Still should get result from readability or BeautifulSoup
            result = await loader.extract_with_multiple_methods("https://example.com")
            
            # Verify the extraction
            assert "title" in result
            assert "text" in result

# Tests for NewsLoader

@pytest.mark.asyncio
async def test_news_loader_fetch_google_news(mock_news_rss, mock_httpx_client, mock_trafilatura):
    """Test the NewsLoader.fetch_google_news method"""
    # Setup mock for extract_with_multiple_methods
    with patch('agents.scraping_agent.document_loaders.NewsLoader.extract_with_multiple_methods') as mock_extract:
        mock_extract.return_value = {
            'title': 'Test Article',
            'text': 'This is test content for the article',
            'source': 'https://example.com/news/1',
            'author': 'Test Author',
            'date': datetime.now()
        }
        
        loader = NewsLoader()
        articles = await loader.fetch_google_news("test topic", 2)
        
        # Verify the results
        assert len(articles) == 2
        assert isinstance(articles[0], NewsArticle)
        assert articles[0].title == 'Test Article 1'  # From RSS feed
        assert articles[0].source == 'Google News'
        assert "test content" in articles[0].body

@pytest.mark.asyncio
async def test_news_loader_fetch_yahoo_finance_news(mock_yfinance, mock_httpx_client, mock_trafilatura):
    """Test the NewsLoader.fetch_yahoo_finance_news method"""
    # Setup mock for extract_with_multiple_methods
    with patch('agents.scraping_agent.document_loaders.NewsLoader.extract_with_multiple_methods') as mock_extract:
        mock_extract.return_value = {
            'title': 'Finance News',
            'text': 'Financial news content',
            'source': 'https://finance.example.com/news/1',
            'author': 'Finance Reporter',
            'date': datetime.now()
        }
        
        loader = NewsLoader()
        articles = await loader.fetch_yahoo_finance_news("AAPL", 2)
        
        # Verify the results
        assert len(articles) == 2
        assert isinstance(articles[0], NewsArticle)
        assert articles[0].title == 'Test Company News'  # From mocked yfinance
        assert articles[0].source == 'Test Publisher'
        assert "Financial news content" in articles[0].body

# Tests for SECFilingLoader

@pytest.mark.asyncio
async def test_sec_filing_process_filing(mock_httpx_client, mock_trafilatura):
    """Test the SECFilingLoader._process_filing method"""
    # Setup mock for extract_with_multiple_methods
    with patch('agents.scraping_agent.document_loaders.SECFilingLoader.extract_with_multiple_methods') as mock_extract:
        mock_extract.return_value = {
            'title': 'SEC Filing',
            'text': 'FORM 10-K\nCOMPANY CONFORMED NAME: Test Corp\nCONFORMED PERIOD OF REPORT: 20250421',
            'source': 'https://sec.gov/example',
            'date': datetime.now()
        }
        
        loader = SECFilingLoader()
        filing = await loader._process_filing("https://sec.gov/example")
        
        # Verify the results
        assert isinstance(filing, FilingResponse)
        assert filing.title == 'SEC Filing'
        assert "FORM 10-K" in filing.body
        assert filing.filing_type == 'FORM 10-K'
        assert filing.company == 'Test Corp'

# Tests for CompanyProfileLoader

@pytest.mark.asyncio
async def test_company_profile_loader(mock_yfinance):
    """Test the CompanyProfileLoader.fetch_company_profile method"""
    loader = CompanyProfileLoader()
    profile = await loader.fetch_company_profile("TEST")
    
    # Verify the results
    assert profile.symbol == "TEST"
    assert profile.name == "Test Company Inc."
    assert profile.sector == "Technology"
    assert profile.industry == "Software"
    assert profile.market_cap == 1000000000
    assert profile.price == 150.75

# Tests for EarningsLoader

@pytest.mark.asyncio
async def test_earnings_loader(mock_yfinance):
    """Test the EarningsLoader.fetch_latest_earnings method"""
    loader = EarningsLoader()
    earnings = await loader.fetch_latest_earnings("TEST")
    
    # Verify the results
    assert earnings.symbol == "TEST"
    assert earnings.company_name == "Test Company Inc."
    assert earnings.eps_estimate == 2.75
    assert earnings.eps_actual == 2.5
    assert earnings.revenue_estimate == 1050000000


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
