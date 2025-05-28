"""
Enhanced document loaders for the Scraping Agent microservice.
Provides multiple strategies for legally scraping financial news and data.
"""
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import urllib.parse
import re

# HTTP and networking
import httpx
import aiohttp
import aiofiles

# Data parsing and extraction
from bs4 import BeautifulSoup
import feedparser
import newspaper
from newspaper import Article
import trafilatura
from readability import Document
import markdown
import yfinance as yf
from secedgar import filings, CompanyFilings
from secedgar.utils import get_quarter

# Error handling
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, RetryError
)

# Models
from .models import (
    NewsArticle, NewsResponse, FilingResponse,
    CompanyNewsResponse, EarningsResponse, MarketNewsResponse,
    FinancialReportResponse, CompanyProfileResponse
)
from .config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)


class ScrapingError(Exception):
    """Base exception for scraping errors."""
    pass


class RateLimitError(ScrapingError):
    """Exception raised when rate limits are encountered."""
    pass


class ContentExtractionError(ScrapingError):
    """Exception raised when content extraction fails."""
    pass


class DocumentLoader:
    """Base class for document loaders."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.user_agent = settings.USER_AGENT
        
    async def get_content(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {"User-Agent": self.user_agent}
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            raise ScrapingError(f"Failed to fetch content: {str(e)}")
    
    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ScrapingError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def extract_with_multiple_methods(self, url: str) -> Dict[str, Any]:
        """
        Extract content using multiple extraction methods and return the best result.
        
        Returns:
            Dictionary with title, text, metadata
        """
        html = await self.get_content(url)
        results = {}
        errors = []
        
        # Method 1: Trafilatura (usually best quality)
        try:
            trafilatura_result = trafilatura.extract(
                html, 
                include_comments=False, 
                include_tables=True,
                include_links=True,
                output_format="json"
            )
            if trafilatura_result:
                import json
                traf_data = json.loads(trafilatura_result)
                results["trafilatura"] = {
                    "title": traf_data.get("title", ""),
                    "text": traf_data.get("text", ""),
                    "author": traf_data.get("author", ""),
                    "date": traf_data.get("date", ""),
                    "source": url
                }
        except Exception as e:
            errors.append(f"Trafilatura error: {str(e)}")
        
        # Method 2: Newspaper3k
        try:
            article = Article(url)
            article.download(input_html=html)
            article.parse()
            results["newspaper"] = {
                "title": article.title,
                "text": article.text,
                "author": ", ".join(article.authors),
                "date": article.publish_date,
                "source": url
            }
        except Exception as e:
            errors.append(f"Newspaper3k error: {str(e)}")
        
        # Method 3: Readability
        try:
            doc = Document(html)
            results["readability"] = {
                "title": doc.title(),
                "text": doc.summary(),
                "author": "",
                "date": None,
                "source": url
            }
        except Exception as e:
            errors.append(f"Readability error: {str(e)}")
        
        # Method 4: BeautifulSoup fallback
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into chunks
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Join chunks with newlines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            title = soup.title.string if soup.title else "Unknown Title"
            
            results["beautifulsoup"] = {
                "title": title,
                "text": text[:5000],  # Limit to first 5000 chars
                "author": "",
                "date": None,
                "source": url
            }
        except Exception as e:
            errors.append(f"BeautifulSoup error: {str(e)}")
        
        # Choose the best result
        if "trafilatura" in results:
            best = "trafilatura"
        elif "newspaper" in results:
            best = "newspaper"
        elif "readability" in results:
            best = "readability"
        elif "beautifulsoup" in results:
            best = "beautifulsoup"
        else:
            error_msg = "; ".join(errors)
            raise ContentExtractionError(f"All extraction methods failed: {error_msg}")
        
        return results[best]


class NewsLoader(DocumentLoader):
    """Loader for news articles with multiple source support."""
    
    async def fetch_google_news(self, topic: str, limit: int = 5) -> List[NewsArticle]:
        """Fetch news from Google News via RSS."""
        articles = []
        
        # Format the query for Google News
        encoded_topic = urllib.parse.quote_plus(topic)
        rss_url = f"https://news.google.com/rss/search?q={encoded_topic}"
        
        try:
            # Get RSS content
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    rss_url, 
                    headers={"User-Agent": self.user_agent}
                )
                response.raise_for_status()
            
            # Parse with feedparser
            feed = feedparser.parse(response.text)
            
            # Process entries
            for i, entry in enumerate(feed.entries[:limit]):
                if i >= limit:
                    break
                    
                title = entry.title
                link = entry.link
                published = entry.get('published')
                
                try:
                    # Extract full article content
                    content = await self.extract_with_multiple_methods(link)
                    
                    # Parse date if available
                    pub_date = None
                    if published:
                        try:
                            # Feedparser returns a time struct
                            pub_date = datetime(*published[:6])
                        except:
                            # Try to parse as string
                            try:
                                pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                            except:
                                pass
                    
                    # Use extracted date if we couldn't get from feed
                    if not pub_date and content.get('date'):
                        pub_date = content.get('date')
                    
                    # Create article
                    article = NewsArticle(
                        title=title,
                        body=content.get('text', '')[:2000],  # Limit length
                        url=link,
                        source="Google News",
                        published_date=pub_date
                    )
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Error extracting article {link}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error fetching Google News: {str(e)}")
            raise ScrapingError(f"Google News fetch failed: {str(e)}")
        
        return articles
    
    async def fetch_yahoo_finance_news(self, symbol: str, limit: int = 5) -> List[NewsArticle]:
        """Fetch news about a company from Yahoo Finance."""
        articles = []
        
        try:
            # Run yfinance in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            
            # Get news
            news_items = await loop.run_in_executor(None, lambda: ticker.news)
            
            # Process news items
            for i, item in enumerate(news_items[:limit]):
                if i >= limit:
                    break
                
                title = item.get('title', f"News about {symbol}")
                link = item.get('link', '')
                
                # Convert timestamp to datetime
                pub_date = None
                if 'providerPublishTime' in item:
                    pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                
                try:
                    # If we have a valid URL, extract content
                    if link and link.startswith('http'):
                        content = await self.extract_with_multiple_methods(link)
                        article_text = content.get('text', '')
                    else:
                        article_text = item.get('summary', '')
                    
                    # Create article
                    article = NewsArticle(
                        title=title,
                        body=article_text[:2000],  # Limit length
                        url=link,
                        source=item.get('publisher', 'Yahoo Finance'),
                        published_date=pub_date
                    )
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Error extracting Yahoo Finance article {link}: {str(e)}")
                    # Still create an article with available data
                    article = NewsArticle(
                        title=title,
                        body=item.get('summary', '')[:2000],
                        url=link,
                        source=item.get('publisher', 'Yahoo Finance'),
                        published_date=pub_date
                    )
                    articles.append(article)
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {str(e)}")
            raise ScrapingError(f"Yahoo Finance news fetch failed: {str(e)}")
        
        return articles
    
    async def fetch_market_news(self, limit: int = 5) -> List[NewsArticle]:
        """Fetch general market news from multiple sources."""
        sources = [
            "https://finance.yahoo.com/news/",
            "https://www.marketwatch.com/latest-news",
            "https://www.investing.com/news/latest-news"
        ]
        
        articles = []
        tasks = []
        
        for source_url in sources:
            task = asyncio.create_task(self._fetch_from_source(source_url, limit // len(sources)))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error fetching from source: {str(result)}")
            else:
                articles.extend(result)
        
        # Sort by publication date if available
        articles.sort(key=lambda x: x.published_date if x.published_date else datetime.min, reverse=True)
        
        return articles[:limit]
    
    async def _fetch_from_source(self, url: str, limit: int) -> List[NewsArticle]:
        """Helper method to fetch articles from a specific financial news source."""
        articles = []
        source_name = url.split('/')[2]
        
        try:
            html = await self.get_content(url)
            
            # Use newspaper to extract article links
            news_site = newspaper.build(
                url=url,
                memoize_articles=False,
                language='en',
                fetch_images=False,
                request_timeout=self.timeout
            )
            
            # Get article URLs
            article_urls = []
            for article in news_site.articles[:limit*3]:  # Get more than needed in case some fail
                article_urls.append(article.url)
            
            # Process each article
            for i, article_url in enumerate(article_urls):
                if i >= limit:
                    break
                
                try:
                    content = await self.extract_with_multiple_methods(article_url)
                    
                    article = NewsArticle(
                        title=content.get('title', f"Article from {source_name}"),
                        body=content.get('text', '')[:2000],
                        url=article_url,
                        source=source_name,
                        published_date=content.get('date')
                    )
                    articles.append(article)
                    
                    # Be nice to the server with a small delay
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error extracting article from {source_name}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {str(e)}")
        
        return articles


class SECFilingLoader(DocumentLoader):
    """Loader for SEC filings with enhanced extraction capabilities."""
    
    async def fetch_company_filings(self, ticker: str, form_type: str = None, limit: int = 5) -> List[FilingResponse]:
        """
        Fetch SEC filings for a company.
        
        Args:
            ticker: Company ticker symbol
            form_type: Optional form type (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to return
            
        Returns:
            List of FilingResponse objects
        """
        filings_list = []
        
        try:
            # Use secedgar to get filings
            loop = asyncio.get_event_loop()
            
            # Determine quarters to fetch (current and previous)
            current_year = datetime.now().year
            current_quarter = get_quarter(datetime.now().month)
            
            quarters = [(current_year, current_quarter)]
            # Add previous quarter
            if current_quarter == 1:
                quarters.append((current_year-1, 4))
            else:
                quarters.append((current_year, current_quarter-1))
            
            # Create filing object with specified form type
            if form_type:
                my_filings = CompanyFilings(ticker, form_type, user_agent=self.user_agent)
            else:
                my_filings = CompanyFilings(ticker, user_agent=self.user_agent)
            
            # Get filing URLs
            filing_urls = []
            for year, quarter in quarters:
                try:
                    # This can be slow, run in executor
                    quarter_filings = await loop.run_in_executor(
                        None, 
                        lambda: my_filings.get_urls(year=year, quarter=quarter)
                    )
                    filing_urls.extend(quarter_filings[:limit])
                except Exception as qe:
                    logger.warning(f"Error getting filings for {ticker} in {year}Q{quarter}: {str(qe)}")
            
            # Limit number of filings
            filing_urls = filing_urls[:limit]
            
            # Process each filing
            for url in filing_urls:
                try:
                    filing = await self._process_filing(url)
                    filings_list.append(filing)
                except Exception as e:
                    logger.warning(f"Error processing filing {url}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {str(e)}")
            raise ScrapingError(f"SEC filing fetch failed: {str(e)}")
        
        return filings_list
    
    async def _process_filing(self, url: str) -> FilingResponse:
        """Process a single SEC filing."""
        try:
            # Extract content
            content = await self.extract_with_multiple_methods(url)
            
            # Try to extract filing metadata
            filing_type = None
            filing_date = None
            company = None
            
            # Common patterns in SEC filings
            text = content.get('text', '')
            
            # Extract form type
            form_match = re.search(r'FORM\s+(10-K|10-Q|8-K|S-1|13F|4|DEF 14A)', text)
            if form_match:
                filing_type = form_match.group(0)
            
            # Extract filing date
            date_match = re.search(r'CONFORMED PERIOD OF REPORT:\s+(\d{8})', text)
            if date_match:
                date_str = date_match.group(1)
                try:
                    filing_date = datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    pass
            
            # Extract company name
            company_match = re.search(r'COMPANY CONFORMED NAME:\s+(.+?)(?:\n|$)', text)
            if company_match:
                company = company_match.group(1).strip()
            
            # Create filing response
            return FilingResponse(
                source=url,
                timestamp=datetime.now(),
                title=content.get('title', f"Filing from {url}"),
                body=text[:10000],  # Limit to avoid extremely large responses
                filing_type=filing_type,
                filing_date=filing_date,
                company=company
            )
            
        except Exception as e:
            logger.error(f"Error processing SEC filing {url}: {str(e)}")
            raise ContentExtractionError(f"Failed to process SEC filing: {str(e)}")


class CompanyProfileLoader(DocumentLoader):
    """Loader for company profile and financial data."""
    
    async def fetch_company_profile(self, symbol: str) -> CompanyProfileResponse:
        """
        Fetch a company profile with key information.
        
        Args:
            symbol: Company ticker symbol
            
        Returns:
            CompanyProfileResponse with company data
        """
        try:
            # Use yfinance to get company info
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            
            # Get company info
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            if not info:
                raise ValueError(f"No information found for {symbol}")
                
            # Extract relevant information
            profile = CompanyProfileResponse(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                description=info.get('longBusinessSummary', ''),
                sector=info.get('sector', ''),
                industry=info.get('industry', ''),
                website=info.get('website', ''),
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE', 0),
                price=info.get('currentPrice', 0),
                employees=info.get('fullTimeEmployees', 0),
                country=info.get('country', ''),
                timestamp=datetime.now()
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {str(e)}")
            raise ScrapingError(f"Company profile fetch failed: {str(e)}")


class EarningsLoader(DocumentLoader):
    """Loader for earnings reports and transcripts."""
    
    async def fetch_latest_earnings(self, symbol: str) -> EarningsResponse:
        """
        Fetch the latest earnings data for a company.
        
        Args:
            symbol: Company ticker symbol
            
        Returns:
            EarningsResponse with earnings data
        """
        try:
            # Use yfinance to get earnings info
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            
            # Get earnings data
            calendar = await loop.run_in_executor(None, lambda: ticker.calendar)
            earnings = await loop.run_in_executor(None, lambda: ticker.earnings)
            
            # Check if we have data
            if not earnings or not calendar:
                raise ValueError(f"No earnings data found for {symbol}")
            
            # Get earnings date
            earnings_date = None
            if 'Earnings Date' in calendar:
                earnings_date = calendar['Earnings Date'][0]
            
            # Create earnings response
            response = EarningsResponse(
                symbol=symbol,
                company_name=ticker.info.get('longName', symbol),
                earnings_date=earnings_date,
                eps_estimate=float(calendar.get('EPS Estimate', [0])[0]) if 'EPS Estimate' in calendar else None,
                eps_actual=None,  # Will be filled if available
                revenue_estimate=float(calendar.get('Revenue Estimate', [0])[0]) if 'Revenue Estimate' in calendar else None,
                revenue_actual=None,  # Will be filled if available
                quarter=None,  # Will be determined if possible
                year=datetime.now().year,
                surprise_percent=None,
                transcript=None,
                timestamp=datetime.now()
            )
            
            # Try to get historical earnings data
            if isinstance(earnings, dict) and 'quarterly' in earnings:
                quarterly = earnings['quarterly']
                if not quarterly.empty:
                    # Get the most recent quarter
                    recent = quarterly.iloc[-1]
                    response.eps_actual = float(recent.get('Earnings', 0))
                    response.revenue_actual = float(recent.get('Revenue', 0))
                    
                    # Try to get the quarter from the date
                    if hasattr(recent, 'index') and recent.index:
                        date = recent.index[0] if isinstance(recent.index, list) else recent.index
                        if hasattr(date, 'quarter'):
                            response.quarter = date.quarter
                            response.year = date.year
                        
                    # Calculate surprise if we have both estimate and actual
                    if response.eps_estimate and response.eps_actual:
                        response.surprise_percent = (response.eps_actual - response.eps_estimate) / abs(response.eps_estimate) * 100
            
            # Try to get earnings transcript
            # Note: this isn't directly available via yfinance, would need to scrape from other sources
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {str(e)}")
            raise ScrapingError(f"Earnings fetch failed: {str(e)}")


# Create loader instances for export
news_loader = NewsLoader()
sec_filing_loader = SECFilingLoader()
company_profile_loader = CompanyProfileLoader()
earnings_loader = EarningsLoader()
