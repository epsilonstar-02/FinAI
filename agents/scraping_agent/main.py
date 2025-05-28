"""
Enhanced FastAPI application for the Scraping Agent microservice.
Implements endpoints for news, SEC filings, company profiles, and more.
"""
import logging
import time
from fastapi import FastAPI, HTTPException, Query, Depends, status, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated, List, Optional

# Import enhanced models
from .models import (
    # Request models
    NewsRequest, CompanyNewsRequest, MarketNewsRequest, 
    FilingRequest, CompanyFilingsRequest, CompanyProfileRequest,
    EarningsRequest,
    
    # Response models
    NewsResponse, CompanyNewsResponse, MarketNewsResponse, 
    FilingResponse, CompanyFilingsResponse, CompanyProfileResponse,
    EarningsResponse, FinancialReportResponse
)

# Import enhanced document loaders
from .document_loaders import (
    news_loader, sec_filing_loader, company_profile_loader, earnings_loader,
    ScrapingError, ContentExtractionError, RateLimitError
)
from .config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="Enhanced Scraping Agent",
    description="""Microservice for scraping financial news, SEC filings, company profiles, and more.
    
    All scraping is done using legal, open-source methods with proper rate limiting and caching.
    Data sources include: news sites, SEC EDGAR, Yahoo Finance, and other public financial data repositories.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handlers
@app.exception_handler(ScrapingError)
async def scraping_error_handler(request, exc):
    """Handle scraping-specific errors."""
    logger.error(f"Scraping error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )

@app.exception_handler(RateLimitError)
async def rate_limit_error_handler(request, exc):
    """Handle rate limit errors."""
    logger.warning(f"Rate limit error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all uncaught exceptions."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    
    logger.error(f"Uncaught exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )

# Health check endpoint
@app.get("/health", 
         summary="Health check endpoint",
         description="Returns the health status of the Scraping Agent")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "agent": "Enhanced Scraping Agent",
        "version": "2.0.0",
        "features": ["news", "sec_filings", "company_profiles", "earnings", "market_news"],
        "timestamp": time.time()
    }

# News scraping endpoints

@app.get("/news", 
         response_model=NewsResponse,
         summary="Fetch general news articles",
         description="Fetches news articles for a specified topic")
async def get_news(
    request: NewsRequest = Depends()
):
    """
    Fetch news articles for a specified topic.
    
    Args:
        request: NewsRequest containing topic, limit, and optional source
        
    Returns:
        NewsResponse containing the scraped articles
    """
    # Input validation already handled by Pydantic
    try:
        # Use Google News as default source if not specified
        if request.source == "yahoo" or request.source == "yahoo_finance":
            articles = await news_loader.fetch_yahoo_finance_news(request.topic, request.limit)
        else:  # Default to Google News
            articles = await news_loader.fetch_google_news(request.topic, request.limit)
        
        return NewsResponse(
            source=request.source or "Google News",
            timestamp=time.time(),
            articles=articles,
            query=request.topic
        )
    except Exception as e:
        logger.error(f"Error in get_news: {str(e)}")
        raise HTTPException(status_code=502, detail=f"News scraping failed: {str(e)}")

@app.get("/company/news/{symbol}", 
         response_model=CompanyNewsResponse,
         summary="Fetch company-specific news",
         description="Fetches news articles for a specific company by ticker symbol")
async def get_company_news(
    symbol: str = Path(..., description="Stock ticker symbol"),
    limit: int = Query(5, description="Maximum number of articles", ge=1, le=20)
):
    """
    Fetch news articles for a specific company.
    
    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of articles to return
        
    Returns:
        CompanyNewsResponse containing the scraped articles
    """
    try:
        # Use Yahoo Finance for company-specific news
        articles = await news_loader.fetch_yahoo_finance_news(symbol, limit)
        
        # If we didn't get enough articles, try searching by company name too
        if len(articles) < limit:
            try:
                # Try to get company name
                profile = await company_profile_loader.fetch_company_profile(symbol)
                company_name = profile.name
                
                # Get more news with company name
                name_articles = await news_loader.fetch_google_news(company_name, limit - len(articles))
                articles.extend(name_articles)
            except Exception as name_error:
                logger.warning(f"Error getting company name news: {str(name_error)}")
        
        return CompanyNewsResponse(
            source="Yahoo Finance",
            timestamp=time.time(),
            articles=articles,
            query=symbol,
            symbol=symbol,
            company_name=articles[0].source if articles else None
        )
    except Exception as e:
        logger.error(f"Error in get_company_news: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Company news scraping failed: {str(e)}")

@app.get("/market/news", 
         response_model=MarketNewsResponse,
         summary="Fetch market news",
         description="Fetches general market news from multiple sources")
async def get_market_news(
    request: MarketNewsRequest = Depends()
):
    """
    Fetch general market news from multiple sources.
    
    Args:
        request: MarketNewsRequest with limit and optional category
        
    Returns:
        MarketNewsResponse containing the scraped articles
    """
    try:
        articles = await news_loader.fetch_market_news(request.limit)
        
        return MarketNewsResponse(
            source="Multiple Financial News Sources",
            timestamp=time.time(),
            articles=articles,
            query="market news",
            category=request.category
        )
    except Exception as e:
        logger.error(f"Error in get_market_news: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Market news scraping failed: {str(e)}")

# SEC filing endpoints

@app.post("/filing", 
           response_model=FilingResponse,
           summary="Fetch SEC filing document",
           description="Fetches an SEC filing document from a URL")
async def get_filing(request: FilingRequest):
    """
    Fetch SEC filing document from a URL.
    
    Args:
        request: FilingRequest containing the URL of the SEC filing
        
    Returns:
        FilingResponse containing the scraped filing
    """
    try:
        return await sec_filing_loader._process_filing(str(request.filing_url))
    except Exception as e:
        logger.error(f"Error in get_filing: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Filing scraping failed: {str(e)}")

@app.get("/company/filings/{symbol}", 
         response_model=CompanyFilingsResponse,
         summary="Fetch company SEC filings",
         description="Fetches SEC filings for a specific company by ticker symbol")
async def get_company_filings(
    symbol: str = Path(..., description="Stock ticker symbol"),
    form_type: Optional[str] = Query(None, description="SEC form type (e.g., '10-K', '10-Q', '8-K')"),
    limit: int = Query(5, description="Maximum number of filings", ge=1, le=10)
):
    """
    Fetch SEC filings for a specific company.
    
    Args:
        symbol: Stock ticker symbol
        form_type: Optional SEC form type filter
        limit: Maximum number of filings to return
        
    Returns:
        CompanyFilingsResponse containing the scraped filings
    """
    try:
        filings = await sec_filing_loader.fetch_company_filings(symbol, form_type, limit)
        
        # Try to get company name
        company_name = None
        try:
            profile = await company_profile_loader.fetch_company_profile(symbol)
            company_name = profile.name
        except Exception:
            if filings and filings[0].company:
                company_name = filings[0].company
        
        return CompanyFilingsResponse(
            symbol=symbol,
            company_name=company_name,
            timestamp=time.time(),
            filings=filings
        )
    except Exception as e:
        logger.error(f"Error in get_company_filings: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Company filings scraping failed: {str(e)}")

# Company profile endpoint

@app.get("/company/profile/{symbol}", 
         response_model=CompanyProfileResponse,
         summary="Fetch company profile",
         description="Fetches company profile and basic information by ticker symbol")
async def get_company_profile(
    symbol: str = Path(..., description="Stock ticker symbol")
):
    """
    Fetch company profile and basic information.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        CompanyProfileResponse containing the company profile
    """
    try:
        return await company_profile_loader.fetch_company_profile(symbol)
    except Exception as e:
        logger.error(f"Error in get_company_profile: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Company profile scraping failed: {str(e)}")

# Earnings endpoint

@app.get("/company/earnings/{symbol}", 
         response_model=EarningsResponse,
         summary="Fetch company earnings",
         description="Fetches latest earnings data for a company by ticker symbol")
async def get_company_earnings(
    symbol: str = Path(..., description="Stock ticker symbol")
):
    """
    Fetch latest earnings data for a company.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        EarningsResponse containing the earnings data
    """
    try:
        return await earnings_loader.fetch_latest_earnings(symbol)
    except Exception as e:
        logger.error(f"Error in get_company_earnings: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Earnings scraping failed: {str(e)}")

# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
