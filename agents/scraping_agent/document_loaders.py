# agents/scraping_agent/document_loaders.py
# This will be the sole loader implementation file.
# `client_loader.py` is deprecated and its logic incorporated here or superseded.

import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import urllib.parse
import re
import json

import httpx
from bs4 import BeautifulSoup
import feedparser
import newspaper # newspaper.Article is aliased as NewspaperArticle
from newspaper import Article as NewspaperArticle 
import trafilatura
from readability import Document as ReadabilityDocument # Aliased to avoid conflict

import yfinance as yf
# secedgar needs careful handling for async and its own user-agent requirements
from secedgar import filings, CompanyFilings 
from secedgar.utils import get_quarter, SymbolNotFoundError

from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, RetryError
)

from .models import NewsArticle, FilingResponse, CompanyProfileResponse, EarningsResponse
from .config import settings

logger = logging.getLogger(__name__)

# Custom Exceptions
class ScrapingError(Exception): pass
class RateLimitError(ScrapingError): pass
class ContentExtractionError(ScrapingError): pass
class SourceUnavailableError(ScrapingError): pass


class DocumentLoader:
    """Base class for document loaders with common utilities."""
    
    def __init__(self, timeout: int = settings.TIMEOUT):
        self.timeout = timeout
        self.user_agent = settings.USER_AGENT
        
    async def _get_html_content(self, url: str) -> str:
        """Fetch HTML content from a URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, http2=True) as client: # Enable HTTP/2
                headers = {"User-Agent": self.user_agent, "Accept-Language": "en-US,en;q=0.9"}
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                # Detect encoding and decode, fallback to utf-8
                content_type = response.headers.get('content-type', '').lower()
                charset = None
                if 'charset=' in content_type:
                    charset = content_type.split('charset=')[-1].split(';')[0].strip()
                
                try:
                    return response.content.decode(charset if charset else response.encoding or 'utf-8')
                except (UnicodeDecodeError, LookupError): # LookupError for invalid encoding name
                    logger.warning(f"Encoding issue for {url}. Trying with utf-8 replacement.")
                    return response.content.decode('utf-8', errors='replace')

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise SourceUnavailableError(f"URL not found (404): {url}") from e
            elif e.response.status_code == 429: # Too Many Requests
                raise RateLimitError(f"Rate limit hit for {url}") from e
            logger.error(f"HTTP error fetching content from {url}: {e.response.status_code}")
            raise ScrapingError(f"HTTP {e.response.status_code} for {url}") from e
        except httpx.RequestError as e: # Network errors
            logger.error(f"Request error fetching content from {url}: {e}")
            raise SourceUnavailableError(f"Network error for {url}: {e}") from e
        except Exception as e: # Other errors
            logger.error(f"Unexpected error fetching content from {url}: {e}", exc_info=True)
            raise ScrapingError(f"Failed to fetch content from {url}: {e}") from e
    
    def _parse_date_flexible(self, date_input: Any) -> Optional[datetime]:
        if not date_input: return None
        if isinstance(date_input, datetime): return date_input
        if isinstance(date_input, date): return datetime.combine(date_input, datetime.min.time())
        if isinstance(date_input, (int, float)): # Assume Unix timestamp
            try: return datetime.fromtimestamp(date_input)
            except ValueError: logger.warning(f"Invalid timestamp for date: {date_input}"); return None

        if isinstance(date_input, str):
            # Try common formats, including ISO 8601 variations
            # Order matters: more specific formats first
            formats_to_try = [
                "%Y-%m-%dT%H:%M:%S.%f%z", # With microseconds and timezone
                "%Y-%m-%dT%H:%M:%S%z",    # Without microseconds, with timezone
                "%Y-%m-%dT%H:%M:%S.%f",   # With microseconds, no timezone (assume UTC or local)
                "%Y-%m-%dT%H:%M:%S",      # Without microseconds, no timezone
                "%a, %d %b %Y %H:%M:%S %Z",# RSS style (e.g., GMT)
                "%a, %d %b %Y %H:%M:%S %z",# RSS style (e.g., +0000)
                "%Y-%m-%d %H:%M:%S",      # Common DB format
                "%Y/%m/%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d",               # Date only
                "%m/%d/%Y",
            ]
            for fmt in formats_to_try:
                try:
                    # Handle 'Z' for UTC explicitly for strptime
                    processed_date_input = date_input.replace("Z", "+0000") if "Z" in date_input and "%z" in fmt else date_input
                    return datetime.strptime(processed_date_input, fmt)
                except ValueError:
                    continue
            logger.warning(f"Could not parse date string with known formats: '{date_input}'")
            return None
        return None # Unrecognized type

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, ScrapingError, SourceUnavailableError)),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10), # min/max wait seconds
        reraise=True # Reraise the exception after retries exhausted
    )
    async def extract_article_data(self, url: str, html_content: Optional[str] = None) -> Dict[str, Any]:
        """Extracts main content, title, date, author from HTML using multiple methods."""
        if not html_content:
            html_content = await self._get_html_content(url)

        extracted_data: Dict[str, Any] = {"title": None, "text": None, "author": None, "date": None, "source": url, "extraction_method": "None"}
        best_text_len = 0

        # Method 1: Trafilatura (often highest quality)
        try:
            # ensure_json=True for direct dict output
            traf_json_str = trafilatura.extract(html_content, url=url, include_comments=False, include_tables=False, output_format='json', ensure_json=True)
            if traf_json_str: # Will be dict if ensure_json=True and successful
                traf_data = traf_json_str # It's already a dict
                current_text = traf_data.get("text", "")
                if len(current_text) > best_text_len:
                    best_text_len = len(current_text)
                    extracted_data.update({
                        "title": traf_data.get("title"), "text": current_text,
                        "author": traf_data.get("author"), "date": self._parse_date_flexible(traf_data.get("date")),
                        "extraction_method": "trafilatura"
                    })
        except Exception as e: logger.debug(f"Trafilatura failed for {url}: {e}")

        # Method 2: Newspaper3k
        try:
            article = NewspaperArticle(url, fetch_images=False, request_timeout=self.timeout)
            article.set_html(html_content) # Use pre-fetched HTML
            article.parse()
            current_text = article.text
            if len(current_text) > best_text_len:
                best_text_len = len(current_text)
                extracted_data.update({
                    "title": article.title, "text": current_text,
                    "author": ", ".join(article.authors) if article.authors else None, 
                    "date": self._parse_date_flexible(article.publish_date),
                    "extraction_method": "newspaper3k"
                })
        except Exception as e: logger.debug(f"Newspaper3k failed for {url}: {e}")

        # Method 3: Readability-LXML
        if len(extracted_data.get("text", "")) < 100 : # Try if others produced little text
            try:
                doc = ReadabilityDocument(html_content)
                current_text = BeautifulSoup(doc.summary(html_partial=True), 'html.parser').get_text(separator='\n', strip=True)
                if len(current_text) > best_text_len:
                    best_text_len = len(current_text)
                    extracted_data.update({
                        "title": doc.title(), "text": current_text,
                        "extraction_method": "readability"
                        # Author/Date not typically extracted by readability
                    })
            except Exception as e: logger.debug(f"Readability failed for {url}: {e}")
        
        # Fallback: BeautifulSoup (simple text extraction) if still no good text
        if len(extracted_data.get("text", "")) < 50 : # Very low threshold
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                for SCRIPT_TAGS in soup(["script", "style", "header", "footer", "nav", "aside"]): SCRIPT_TAGS.decompose()
                # Attempt to find main content area
                main_content_tags = soup.find_all(['main', 'article', {'role': 'main'}, {'class': re.compile(r'(content|main|body|story|article)')}])
                content_text_bs = ""
                if main_content_tags:
                    content_text_bs = "\n".join(tag.get_text(separator='\n', strip=True) for tag in main_content_tags)
                if not content_text_bs or len(content_text_bs) < 50: # If specific tags failed, get all text
                    content_text_bs = soup.get_text(separator='\n', strip=True)

                if len(content_text_bs) > best_text_len: # Compare with potentially empty previous best
                     extracted_data.update({
                        "title": soup.title.string.strip() if soup.title and soup.title.string else None,
                        "text": content_text_bs,
                        "extraction_method": "beautifulsoup"
                    })
            except Exception as e: logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")

        if not extracted_data.get("text"):
            raise ContentExtractionError(f"All content extraction methods failed for {url}. HTML head: {html_content[:500]}")
        
        # If title is still missing, try to get it from soup as a last resort
        if not extracted_data.get("title"):
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                if soup.title and soup.title.string:
                    extracted_data["title"] = soup.title.string.strip()
            except Exception: pass # Ignore if this also fails

        return extracted_data


class NewsLoader(DocumentLoader):
    """Loader for news articles from various sources."""

    async def _process_rss_entry_to_article(self, entry: feedparser.FeedParserDict, source_name: str) -> Optional[NewsArticle]:
        """Helper to process a single feedparser entry into a NewsArticle."""
        title = entry.get("title", "Untitled Article")
        link = entry.get("link")
        if not link: return None

        pub_date: Optional[datetime] = None
        if "published_parsed" in entry and entry.published_parsed:
            try: pub_date = datetime(*entry.published_parsed[:6])
            except TypeError: pass # Handle if published_parsed is not a valid time_struct
        elif "published" in entry:
            pub_date = self._parse_date_flexible(entry.published)
        
        summary = None
        if "summary_detail" in entry and entry.summary_detail:
            summary = BeautifulSoup(entry.summary_detail.get('value',''), 'html.parser').get_text(strip=True)
        elif "summary" in entry:
            summary = BeautifulSoup(entry.get('summary',''), 'html.parser').get_text(strip=True)

        try:
            # Fetch and extract full article content
            article_data = await self.extract_article_data(link)
            body_text = article_data.get("text", summary or "") # Use RSS summary if full text fails
            # Prioritize extracted metadata over RSS metadata if available and better
            final_title = article_data.get("title") or title
            final_pub_date = article_data.get("date") or pub_date
            author = article_data.get("author")

            return NewsArticle(
                title=final_title, body=body_text[:5000], url=link, # Limit body length
                source=source_name, published_date=final_pub_date,
                author=author, summary=summary if summary and len(summary) < len(body_text) else None
            )
        except (ContentExtractionError, SourceUnavailableError) as e:
            logger.warning(f"Failed to extract full content for {link} from {source_name}: {e}. Using summary if available.")
            if summary: # Create article with summary if full extraction failed
                return NewsArticle(
                    title=title, body=summary[:5000], url=link, source=source_name, 
                    published_date=pub_date, summary=summary
                )
        except Exception as e_gen:
            logger.error(f"Unexpected error processing entry {link} from {source_name}: {e_gen}", exc_info=True)
        return None

    async def fetch_google_news(self, topic: str, limit: int = 10) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        encoded_topic = urllib.parse.quote_plus(topic)
        rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            logger.info(f"Fetching Google News RSS for '{topic}' from {rss_url}")
            html_content = await self._get_html_content(rss_url) # Fetch RSS feed content
            feed = feedparser.parse(html_content) # Parse it

            if not feed.entries:
                logger.warning(f"No entries in Google News RSS for '{topic}'.")
                return articles

            tasks = [self._process_rss_entry_to_article(entry, "Google News") for entry in feed.entries[:limit]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, NewsArticle): articles.append(result)
                elif result is not None: logger.warning(f"Error processing Google News entry: {result}")
            
        except SourceUnavailableError as e: logger.error(f"Google News RSS feed unavailable for '{topic}': {e}")
        except Exception as e: logger.error(f"General error fetching Google News for '{topic}': {e}", exc_info=True)
        
        return articles[:limit]


    async def fetch_yahoo_finance_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        try:
            logger.info(f"Fetching Yahoo Finance news for symbol: {symbol}")
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            # ticker.news is a list of dicts
            news_items = await loop.run_in_executor(None, getattr, ticker, 'news') # Robust way to get .news

            if not news_items:
                logger.info(f"No news items from yfinance for {symbol}.")
                return articles

            tasks = []
            for item in news_items[:limit]: # Limit before creating tasks
                link = item.get('link')
                if not link or not link.startswith('http'): continue

                title = item.get('title', f"Article for {symbol}")
                pub_ts = item.get('providerPublishTime')
                pub_date = self._parse_date_flexible(pub_ts) # Handles int timestamp
                publisher = item.get('publisher', 'Yahoo Finance Partner')
                
                tasks.append(self._process_rss_entry_to_article(
                    feedparser.FeedParserDict({"title": title, "link": link, "published_parsed": pub_date.timetuple() if pub_date else None, "summary": item.get("summary")}), 
                    publisher # Pass publisher as source_name
                ))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, NewsArticle): articles.append(result)
                elif result is not None: logger.warning(f"Error processing Yahoo Finance news item: {result}")

        except Exception as e: logger.error(f"Error fetching Yahoo Finance news for {symbol}: {e}", exc_info=True)
        return articles[:limit]


    async def fetch_market_news(self, limit: int = 10, category: Optional[str]=None) -> List[NewsArticle]:
        # Example using a general RSS feed if category doesn't map to a specific source
        # This is a simplified example. Real market news often needs specific site parsers or paid APIs.
        # For now, let's use a generic financial news RSS feed (e.g., from a major publisher if available)
        # or fallback to Google News on a broad topic.
        # If category is "stocks", topic could be "stock market".
        
        topic_for_google = category if category else "general financial market"
        logger.info(f"Fetching general market news (category: {category}) via Google News fallback on topic '{topic_for_google}'.")
        return await self.fetch_google_news(topic_for_google, limit)


class SECFilingLoader(DocumentLoader):
    """Loader for SEC filings."""
    
    async def _process_single_filing_url(self, url: str, symbol: Optional[str]=None, pre_detected_form_type: Optional[str]=None) -> FilingResponse:
        logger.debug(f"Processing SEC filing from URL: {url}")
        try:
            # For SEC HTML, Trafilatura might be too aggressive, Newspaper3k or Readability might be better.
            # Or a custom BS4 parser. For now, using extract_article_data.
            article_data = await self.extract_article_data(url)
            
            text_content = article_data.get("text", "")
            extracted_title = article_data.get("title", f"SEC Filing from {url}")
            extracted_date = article_data.get("date")

            # Refined metadata extraction from text_content
            company_name, filing_type, filing_date_from_text = None, pre_detected_form_type, None

            # Company Name
            company_match = re.search(r'COMPANY CONFORMED NAME:\s*([^\n]+)', text_content, re.IGNORECASE)
            if company_match: company_name = company_match.group(1).strip().title()

            # Filing Type (more robust)
            if not filing_type: # Only if not pre-detected
                form_match = re.search(r'FORM\s+TYPE:\s*([^\n]+)', text_content, re.IGNORECASE) # Exact type
                if not form_match: form_match = re.search(r'ACCESSION NUMBER:.*?CONFORMED SUBMISSION TYPE:\s*([^\n]+)', text_content, re.IGNORECASE | re.DOTALL)
                if form_match: filing_type = form_match.group(1).strip().upper()
                # Remove amendment indicators like /A for primary type
                if filing_type and "/A" in filing_type: filing_type = filing_type.split("/A")[0]


            # Filing Date (Period of Report or Filed As Of Date)
            date_patterns = [
                r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', # YYYYMMDD
                r'FILED AS OF DATE:\s*(\d{8})',
                r'DATE AS OF CHANGE:\s*(\d{8})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, text_content, re.IGNORECASE)
                if date_match:
                    try: filing_date_from_text = datetime.strptime(date_match.group(1), "%Y%m%d")
                    except ValueError: pass
                    if filing_date_from_text: break
            
            final_filing_date = filing_date_from_text or extracted_date # Prioritize text, fallback to general extraction

            return FilingResponse(
                source=url, title=extracted_title, body=text_content[:30000], # Limit SEC filing body
                filing_type=filing_type, filing_date=final_filing_date,
                company=company_name, symbol=symbol
            )
        except ContentExtractionError as e:
            logger.error(f"Content extraction failed for SEC filing {url}: {e}")
            raise # Re-raise to be handled by caller
        except Exception as e:
            logger.error(f"Unexpected error processing SEC filing {url}: {e}", exc_info=True)
            raise ScrapingError(f"Failed to process SEC filing {url}: {e}")


    async def fetch_company_filings(self, ticker: str, form_type: Optional[str] = None, limit: int = 5) -> List[FilingResponse]:
        filings_responses: List[FilingResponse] = []
        sec_user_agent = settings.SEC_API_KEY if settings.SEC_API_KEY and "@" in settings.SEC_API_KEY else "FinAI Scraper your.email@example.com"
        
        # Normalize form_type input
        target_forms = None
        if form_type:
            target_forms = [f.strip().upper() for f in form_type.split(',') if f.strip()]
            if not target_forms: target_forms = None # Treat empty string as None

        logger.info(f"Fetching SEC filings for {ticker}. Forms: {target_forms or 'Any'}. Limit: {limit}. User-Agent: {sec_user_agent[:20]}...")

        # secedgar can be blocking. Use run_in_executor.
        # We need to get URLs first, then process content.
        filing_urls_to_process: List[Tuple[str, str]] = [] # (url, detected_form_type_from_listing)
        
        # Try to get a decent number of recent filing URLs to pick from
        # Fetch for a date range to ensure we get something, e.g., last 2 years
        end_date_secedgar = date.today()
        start_date_secedgar = end_date_secedgar - timedelta(days=365 * 2)

        try:
            # If specific forms are requested, try to fetch them directly.
            # secedgar CompanyFilings takes a single filing_type.
            # If multiple forms, we might need to iterate or fetch all and filter.
            forms_for_secedgar_call = target_forms if target_forms and len(target_forms) == 1 else None
            if forms_for_secedgar_call:
                form_to_get = forms_for_secedgar_call[0]
                logger.debug(f"secedgar: Fetching form {form_to_get} for {ticker}")
                # This needs to be wrapped in run_in_executor
                def get_specific_filings():
                    cf = CompanyFilings(
                        cik_lookup=ticker, filing_type=form_to_get,
                        start_date=start_date_secedgar, end_date=end_date_secedgar,
                        user_agent=sec_user_agent, count=limit * 2 # Fetch more to filter later if needed
                    )
                    return cf.get_urls() # Returns dict: {TICKER: [urls]} or {FORM_TYPE: [urls]}

                urls_dict = await asyncio.get_event_loop().run_in_executor(None, get_specific_filings)
                # The key in urls_dict can be inconsistent based on secedgar version/params
                retrieved_urls = urls_dict.get(ticker.upper(), urls_dict.get(form_to_get, []))
                for url in retrieved_urls: filing_urls_to_process.append((url, form_to_get))
            else: # Fetch multiple forms or all recent forms
                logger.debug(f"secedgar: Fetching recent filings (all types or {target_forms}) for {ticker}")
                def get_all_recent_filings():
                    cf = CompanyFilings(
                        cik_lookup=ticker, filing_type=None, # None fetches common types
                        start_date=start_date_secedgar, end_date=end_date_secedgar,
                        user_agent=sec_user_agent, count= (limit * 5) if not target_forms else (limit * len(target_forms) * 2) # Fetch more if filtering
                    )
                    return cf.get_urls() # Returns dict {FORM_TYPE: [urls]}
                
                urls_by_type_dict = await asyncio.get_event_loop().run_in_executor(None, get_all_recent_filings)
                for form, urls in urls_by_type_dict.items():
                    if not target_forms or form.upper() in target_forms: # Filter if target_forms is specified
                        for url in urls: filing_urls_to_process.append((url, form))
            
            # Deduplicate URLs while preserving order somewhat (last seen wins for form type if duplicate URL)
            # And sort by presumed recency (secedgar usually returns recent first, but explicit sort is safer if dates were part of metadata)
            # For now, simple deduplication on URL:
            seen_urls = set()
            unique_filing_urls = []
            for url, detected_form in reversed(filing_urls_to_process): # Process from potentially older to newer if list was mixed
                if url not in seen_urls:
                    unique_filing_urls.append((url, detected_form))
                    seen_urls.add(url)
            filing_urls_to_process = list(reversed(unique_filing_urls)) # Restore somewhat original recency

        except SymbolNotFoundError:
            logger.warning(f"Ticker {ticker} not found by secedgar.")
            return [] # Return empty list if ticker not found
        except RetryError as e_retry: # from tenacity if internal retries in secedgar failed
            logger.error(f"secedgar failed after retries for {ticker}: {e_retry}")
            raise SourceUnavailableError(f"SEC EDGAR source failed for {ticker} after retries.")
        except Exception as e_secedgar:
            logger.error(f"Error fetching filing URLs via secedgar for {ticker}: {e_secedgar}", exc_info=True)
            raise ScrapingError(f"Failed to get filing list for {ticker}: {e_secedgar}")

        if not filing_urls_to_process:
            logger.info(f"No SEC filing URLs found for {ticker} with criteria: forms={target_forms}")
            return []

        # Process content for the top 'limit' URLs (or fewer if not enough found)
        tasks = [self._process_single_filing_url(url, ticker, detected_form) for url, detected_form in filing_urls_to_process[:limit*2]] # Process a bit more for filtering
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, FilingResponse):
                # Final check if we fetched all and need to filter by target_forms
                if target_forms and result.filing_type and result.filing_type not in target_forms:
                    continue # Skip if form type doesn't match
                filings_responses.append(result)
            elif result is not None: # An exception was returned
                logger.warning(f"Error processing one SEC filing for {ticker}: {result}")
        
        # Sort by filing date (most recent first), handling None dates
        filings_responses.sort(key=lambda f: f.filing_date if f.filing_date else datetime.min, reverse=True)
        return filings_responses[:limit]


class CompanyProfileLoader(DocumentLoader):
    """Loader for company profile and basic financial data from yfinance."""
    async def fetch_company_profile(self, symbol: str) -> CompanyProfileResponse:
        try:
            logger.info(f"Fetching company profile for symbol: {symbol}")
            loop = asyncio.get_event_loop()
            # yfinance.Ticker can be blocking on network for first call
            ticker_obj = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            # ticker_obj.info is a blocking call
            info_dict = await loop.run_in_executor(None, getattr, ticker_obj, 'info')

            if not info_dict or not info_dict.get('symbol'): # 'symbol' is a good check for valid data
                raise SourceUnavailableError(f"Incomplete or no data from yfinance for {symbol} profile.")

            def _safe_get_float(key) -> Optional[float]:
                val = info_dict.get(key)
                return float(val) if isinstance(val, (int, float)) else None
            def _safe_get_int(key) -> Optional[int]:
                val = info_dict.get(key)
                return int(val) if isinstance(val, (int, float)) else (int(val) if isinstance(val, str) and val.isdigit() else None)

            return CompanyProfileResponse(
                symbol=info_dict.get('symbol', symbol).upper(),
                name=info_dict.get('longName') or info_dict.get('shortName'),
                description=info_dict.get('longBusinessSummary'),
                sector=info_dict.get('sector'),
                industry=info_dict.get('industry') or info_dict.get('industryDisp'),
                website=info_dict.get('website'),
                market_cap=_safe_get_float('marketCap'),
                pe_ratio=_safe_get_float('trailingPE') or _safe_get_float('forwardPE'),
                price=_safe_get_float('currentPrice') or _safe_get_float('regularMarketPreviousClose'),
                employees=_safe_get_int('fullTimeEmployees'),
                country=info_dict.get('country'),
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol} from yfinance: {e}", exc_info=True)
            raise ScrapingError(f"yfinance profile fetch failed for {symbol}: {e}")


class EarningsLoader(DocumentLoader):
    """Loader for company earnings data from yfinance."""
    async def fetch_latest_earnings(self, symbol: str) -> EarningsResponse:
        try:
            logger.info(f"Fetching earnings data for symbol: {symbol}")
            loop = asyncio.get_event_loop()
            ticker_obj = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))

            # Fetch info, calendar, and earnings data
            # These are blocking calls
            info_dict = await loop.run_in_executor(None, getattr, ticker_obj, 'info')
            calendar_df = await loop.run_in_executor(None, getattr, ticker_obj, 'calendar') # Often a DataFrame
            # .earnings can be a DataFrame or Dict of DataFrames
            earnings_hist_data = await loop.run_in_executor(None, getattr, ticker_obj, 'earnings') 

            company_name = info_dict.get('longName') or info_dict.get('shortName', symbol)

            earnings_date_dt: Optional[datetime] = None
            eps_estimate_val: Optional[float] = None
            revenue_estimate_val: Optional[float] = None

            if calendar_df is not None and not calendar_df.empty and 'Earnings Date' in calendar_df.columns:
                # calendar_df usually has 'Earnings Date' as index or column
                # Assuming 'Earnings Date' is a column with Timestamp objects
                # Take the first available date if multiple are listed (e.g., range)
                earnings_date_series = calendar_df['Earnings Date']
                if not earnings_date_series.empty:
                    # Convert to datetime if it's pandas Timestamp or similar
                    raw_date = earnings_date_series.iloc[0]
                    if pd.notna(raw_date) and hasattr(raw_date, 'to_pydatetime'):
                        earnings_date_dt = raw_date.to_pydatetime()
                    elif isinstance(raw_date, datetime):
                        earnings_date_dt = raw_date
                
                if 'EPS Estimate' in calendar_df.columns: eps_estimate_val = calendar_df['EPS Estimate'].iloc[0]
                if 'Revenue Estimate' in calendar_df.columns: revenue_estimate_val = calendar_df['Revenue Estimate'].iloc[0]
            
            # Try to parse actuals from earnings history
            eps_actual_val, revenue_actual_val = None, None
            report_year, report_quarter = None, None

            quarterly_earnings_df = None
            if isinstance(earnings_hist_data, pd.DataFrame) and not earnings_hist_data.empty:
                quarterly_earnings_df = earnings_hist_data # If it's directly the quarterly data
            elif isinstance(earnings_hist_data, dict) and 'quarterly' in earnings_hist_data and not earnings_hist_data['quarterly'].empty:
                quarterly_earnings_df = earnings_hist_data['quarterly']
            elif isinstance(earnings_hist_data, dict) and 'Quarterly' in earnings_hist_data and not earnings_hist_data['Quarterly'].empty: # yfinance key casing
                quarterly_earnings_df = earnings_hist_data['Quarterly']

            if quarterly_earnings_df is not None and not quarterly_earnings_df.empty:
                latest_actuals = quarterly_earnings_df.sort_index(ascending=False).iloc[0]
                eps_actual_val = latest_actuals.get('ReportedEPS', latest_actuals.get('EPS'))
                revenue_actual_val = latest_actuals.get('Revenue')
                
                # Year/Quarter from index if it's a DatetimeIndex
                if isinstance(quarterly_earnings_df.index, pd.DatetimeIndex):
                    report_dt_idx = quarterly_earnings_df.index[0]
                    report_year = report_dt_idx.year
                    report_quarter = report_dt_idx.quarter
                # If index is like "3Q2023"
                elif isinstance(quarterly_earnings_df.index, pd.Index) and isinstance(quarterly_earnings_df.index[0], str):
                     match_period = re.match(r"(\d)Q(\d{4})", quarterly_earnings_df.index[0])
                     if match_period:
                         report_quarter = int(match_period.group(1))
                         report_year = int(match_period.group(2))


            surprise_pct_val = None
            if eps_actual_val is not None and eps_estimate_val is not None and abs(eps_estimate_val) > 1e-6:
                surprise_pct_val = ((eps_actual_val - eps_estimate_val) / abs(eps_estimate_val)) * 100

            return EarningsResponse(
                symbol=symbol.upper(), company_name=company_name,
                earnings_date=earnings_date_dt,
                eps_estimate=float(eps_estimate_val) if pd.notna(eps_estimate_val) else None,
                eps_actual=float(eps_actual_val) if pd.notna(eps_actual_val) else None,
                revenue_estimate=float(revenue_estimate_val) if pd.notna(revenue_estimate_val) else None,
                revenue_actual=float(revenue_actual_val) if pd.notna(revenue_actual_val) else None,
                quarter=report_quarter, year=report_year,
                surprise_percent=float(surprise_pct_val) if pd.notna(surprise_pct_val) else None,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol} from yfinance: {e}", exc_info=True)
            raise ScrapingError(f"yfinance earnings fetch failed for {symbol}: {e}")

# Instantiate loaders for export
news_loader = NewsLoader()
sec_filing_loader = SECFilingLoader()
company_profile_loader = CompanyProfileLoader()
earnings_loader = EarningsLoader()