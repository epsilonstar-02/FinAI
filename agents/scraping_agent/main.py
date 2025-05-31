# agents/scraping_agent/main.py
# Updated to use consolidated loaders and refined error handling.

import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Depends, status, Path as FastApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional 

from .models import (
    NewsRequest, 
    NewsResponse, CompanyNewsResponse, MarketNewsResponse, 
    FilingRequest, FilingResponse, CompanyFilingsResponse, 
    CompanyProfileResponse, EarningsResponse
)
from .document_loaders import (
    news_loader, sec_filing_loader, company_profile_loader, earnings_loader,
    ScrapingError, ContentExtractionError, RateLimitError, SourceUnavailableError
)
from .config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scraping Agent", # Simplified title
    description="Microservice for scraping financial news, SEC filings, company data.",
    version="2.1.0", # Incremented for this refactor
    docs_url="/docs", redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
    allow_methods=["*"], allow_headers=["*"],
)

# Exception Handlers
@app.exception_handler(SourceUnavailableError)
async def source_unavailable_handler(request, exc: SourceUnavailableError):
    logger.warning(f"Source unavailable for {request.url.path}: {exc}")
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request, exc: RateLimitError):
    logger.warning(f"Rate limit hit processing {request.url.path}: {exc}")
    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": str(exc)})

@app.exception_handler(ContentExtractionError)
async def content_extraction_handler(request, exc: ContentExtractionError):
    logger.error(f"Content extraction failed for {request.url.path}: {exc}")
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": f"Failed to extract content: {exc}"})

@app.exception_handler(ScrapingError) # General scraping errors
async def scraping_error_handler(request, exc: ScrapingError):
    logger.error(f"A scraping error occurred for {request.url.path}: {exc}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": f"Scraping operation failed: {exc}"})

@app.exception_handler(Exception) # Catch-all for unexpected
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unexpected error for {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "An unexpected internal server error occurred."})


# Endpoints
TAG_UTILITY = "Utility"
TAG_NEWS = "News Scraping"
TAG_FILINGS = "SEC Filings"
TAG_COMPANY_DATA = "Company Data"

@app.get("/health", summary="Health Check", tags=[TAG_UTILITY])
async def health_check_endpoint():
    return {"status": "ok", "agent": "Scraping Agent", "version": app.version, "timestamp": datetime.utcnow()}

@app.get("/news", response_model=NewsResponse, summary="General News by Topic", tags=[TAG_NEWS])
async def get_general_news(request: NewsRequest = Depends()):
    # Default to Google News. If source=yahoo specified, it implies topic is a symbol for yahoo_finance_news.
    # However, /company/news/{symbol} is better for symbol-specific Yahoo news.
    # This endpoint is best for topic-based Google News.
    if request.source and request.source.lower() in ["yahoo", "yahoo_finance"]:
        logger.info(f"Received request for Yahoo news with topic '{request.topic}'. Consider /company/news for symbol-specific news.")
        # This might not yield good results if topic isn't a symbol.
        articles = await news_loader.fetch_yahoo_finance_news(request.topic, request.limit)
        source_used = "Yahoo Finance (via Topic)"
    else:
        if request.source and request.source.lower() not in ["google", "google_news"]:
            logger.warning(f"Unsupported news source '{request.source}' for general news, defaulting to Google News.")
        articles = await news_loader.fetch_google_news(request.topic, request.limit)
        source_used = "Google News"
        
    return NewsResponse(source=source_used, articles=articles, query=request.topic)

@app.get("/company/news/{symbol}", response_model=CompanyNewsResponse, summary="Company-Specific News", tags=[TAG_NEWS])
async def get_company_specific_news(
    symbol: str = FastApiPath(..., description="Stock ticker symbol", min_length=1, max_length=10, pattern=r'^[A-Z0-9.\-]+$'),
    limit: int = Query(5, description="Max articles", ge=1, le=20)
):
    # Primary: Yahoo Finance News for the symbol
    articles = await news_loader.fetch_yahoo_finance_news(symbol.upper(), limit)
    
    company_name_val: Optional[str] = None
    # Try to get company name for richer response
    try: profile = await company_profile_loader.fetch_company_profile(symbol.upper()); company_name_val = profile.name
    except ScrapingError: logger.debug(f"Profile fetch failed for {symbol}, company name may be missing in news response.")

    # Fallback: If not enough articles from Yahoo, try Google News with symbol/company name
    if len(articles) < limit:
        logger.info(f"Yahoo News for {symbol} returned {len(articles)} articles (limit {limit}). Trying Google News fallback.")
        google_topic = company_name_val if company_name_val else symbol.upper()
        try:
            google_articles = await news_loader.fetch_google_news(google_topic, limit - len(articles))
            existing_urls = {art.url for art in articles}
            for g_art in google_articles:
                if g_art.url not in existing_urls: articles.append(g_art); existing_urls.add(g_art.url)
        except Exception as e_gn_fb: logger.warning(f"Google News fallback for {symbol} failed: {e_gn_fb}")
    
    return CompanyNewsResponse(
        source="Yahoo Finance / Google News", articles=articles[:limit], 
        query=symbol, symbol=symbol.upper(), company_name=company_name_val
    )

@app.get("/market/news", response_model=MarketNewsResponse, summary="General Market News", tags=[TAG_NEWS])
async def get_general_market_news(request: MarketNewsRequest = Depends()):
    articles = await news_loader.fetch_market_news(request.limit, request.category)
    query_str = "general market news" + (f", category: {request.category}" if request.category else "")
    return MarketNewsResponse(source="Aggregated Market Sources", articles=articles, query=query_str, category=request.category)

@app.post("/filing/by-url", response_model=FilingResponse, summary="Fetch Single SEC Filing by URL", tags=[TAG_FILINGS])
async def get_single_filing_by_url(request: FilingRequest): # Renamed endpoint for clarity
    # sec_filing_loader._process_single_filing_url is the correct method here
    return await sec_filing_loader._process_single_filing_url(str(request.filing_url))

@app.get("/company/filings/{symbol}", response_model=CompanyFilingsResponse, summary="Company SEC Filings", tags=[TAG_FILINGS])
async def get_company_sec_filings(
    symbol: str = FastApiPath(..., description="Stock ticker symbol", min_length=1, max_length=10, pattern=r'^[A-Z0-9.\-]+$'),
    form_type: Optional[str] = Query(None, description="SEC form types (e.g., '10-K,8-K')"),
    limit: int = Query(5, description="Max filings", ge=1, le=15)
):
    filings = await sec_filing_loader.fetch_company_filings(symbol.upper(), form_type, limit)
    
    company_name_val: Optional[str] = None
    if filings and filings[0].company: company_name_val = filings[0].company
    else: # Try profile lookup for company name
        try: profile = await company_profile_loader.fetch_company_profile(symbol.upper()); company_name_val = profile.name
        except ScrapingError: logger.debug(f"Profile fetch failed for {symbol}, company name may be missing in filings response.")
        
    return CompanyFilingsResponse(symbol=symbol.upper(), company_name=company_name_val, filings=filings)

@app.get("/company/profile/{symbol}", response_model=CompanyProfileResponse, summary="Company Profile", tags=[TAG_COMPANY_DATA])
async def get_company_profile_data(
    symbol: str = FastApiPath(..., description="Stock ticker symbol", min_length=1, max_length=10, pattern=r'^[A-Z0-9.\-]+$')
):
    return await company_profile_loader.fetch_company_profile(symbol.upper())

@app.get("/company/earnings/{symbol}", response_model=EarningsResponse, summary="Company Earnings Data", tags=[TAG_COMPANY_DATA])
async def get_company_earnings_data(
    symbol: str = FastApiPath(..., description="Stock ticker symbol", min_length=1, max_length=10, pattern=r'^[A-Z0-9.\-]+$')
):
    return await earnings_loader.fetch_latest_earnings(symbol.upper())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agents.scraping_agent.main:app", host=settings.HOST, port=settings.PORT, reload=True, log_level=settings.LOG_LEVEL.lower())