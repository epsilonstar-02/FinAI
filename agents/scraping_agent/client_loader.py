"""
Document loaders for the Scraping Agent microservice.
Implements various document loaders for different sources.
"""
import logging
from datetime import datetime
from typing import List, Optional
import httpx
from langchain.document_loaders import UnstructuredURLLoader
from bs4 import BeautifulSoup
from fastapi import HTTPException

from .models import NewsResponse, NewsArticle, FilingResponse
from .config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

async def fetch_news_loader(topic: str, limit: int) -> NewsResponse:
    """
    Fetch news articles about a specific topic.
    
    Args:
        topic: The topic to search for news about
        limit: Maximum number of articles to return
        
    Returns:
        NewsResponse containing the scraped articles
    """
    try:
        # Google News URL for the specified topic
        google_news_url = f"https://news.google.com/rss/search?q={topic}"
        
        logger.info(f"Fetching news for topic: {topic}, limit: {limit}")
        
        # First try to fetch RSS feed for structured data
        async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
            response = await client.get(google_news_url, 
                                      headers={"User-Agent": settings.USER_AGENT})
            response.raise_for_status()
            
        # Parse the RSS feed with BeautifulSoup
        soup = BeautifulSoup(response.text, 'xml')
        items = soup.find_all('item')
        
        # Limit the number of articles
        items = items[:limit]
        
        # Extract article URLs
        article_urls = []
        for item in items:
            title = item.title.text if item.title else "Untitled"
            link = item.link.text if item.link else None
            if link:
                article_urls.append((title, link))
        
        # If we couldn't get URLs from RSS, fall back to UnstructuredURLLoader
        if not article_urls:
            logger.warning("No articles found in RSS feed, falling back to UnstructuredURLLoader")
            loader = UnstructuredURLLoader(
                urls=[google_news_url],
                headers={"User-Agent": settings.USER_AGENT}
            )
            docs = loader.load()
            
            articles = []
            for i, doc in enumerate(docs[:limit]):
                articles.append(NewsArticle(
                    title=f"News about {topic} #{i+1}",
                    body=doc.page_content[:500] if len(doc.page_content) > 500 
                         else doc.page_content,
                    url=f"https://news.google.com/search?q={topic}",
                    source="Google News"
                ))
        else:
            # Process each article URL
            articles = []
            for i, (title, url) in enumerate(article_urls):
                try:
                    # For each article URL, fetch content
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        headers={"User-Agent": settings.USER_AGENT}
                    )
                    article_docs = loader.load()
                    
                    if article_docs:
                        article_content = article_docs[0].page_content
                        # Truncate article content if too long
                        article_content = article_content[:1000] if len(article_content) > 1000 else article_content
                        
                        articles.append(NewsArticle(
                            title=title,
                            body=article_content,
                            url=url,
                            source="Google News"
                        ))
                    
                    # Don't overload the server with requests
                    if i >= limit - 1:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error loading article {url}: {str(e)}")
                    continue
        
        # Create and return response
        return NewsResponse(
            source="Google News",
            timestamp=datetime.now(),
            articles=articles
        )
    except Exception as e:
        logger.error(f"Error in fetch_news_loader: {str(e)}")
        raise HTTPException(status_code=502, detail=f"News loader failed: {str(e)}")

async def fetch_filing_loader(filing_url: str) -> FilingResponse:
    """
    Fetch SEC filing document from a URL.
    
    Args:
        filing_url: URL of the SEC filing to scrape
        
    Returns:
        FilingResponse containing the scraped filing
    """
    try:
        logger.info(f"Fetching filing from URL: {filing_url}")
        
        # Create document loader for the filing URL
        loader = UnstructuredURLLoader(
            urls=[filing_url],
            headers={"User-Agent": settings.USER_AGENT}
        )
        
        # Load documents
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from URL")
        
        # Extract content from the first document
        doc = docs[0]
        content = doc.page_content
        
        # Try to extract title and filing type from content
        title = f"Filing from {filing_url}"
        filing_type = None
        company = None
        filing_date = None
        
        # Parse content to extract metadata
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):  # Look at first 20 lines for metadata
            if "FORM" in line.upper() and any(form in line.upper() for form in ["10-K", "10-Q", "8-K", "S-1"]):
                filing_type = line.strip()
            if "CONFORMED PERIOD OF REPORT" in line.upper() and i+1 < len(lines):
                try:
                    date_str = lines[i+1].strip()
                    filing_date = datetime.strptime(date_str, "%Y%m%d")
                except (ValueError, IndexError):
                    pass
            if "COMPANY CONFORMED NAME" in line.upper() and i+1 < len(lines):
                company = lines[i+1].strip()
        
        # Create and return response
        return FilingResponse(
            source=filing_url,
            timestamp=datetime.now(),
            title=title,
            body=content,
            filing_type=filing_type,
            filing_date=filing_date,
            company=company
        )
    except Exception as e:
        logger.error(f"Error in fetch_filing_loader: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Filing loader failed: {str(e)}")
