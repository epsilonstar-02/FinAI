"""
Main FastAPI application for the Scraping Agent microservice.
Implements endpoints for news and SEC filing scraping.
"""
import logging
import time
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated

from .models import (
    NewsRequest, NewsResponse, FilingRequest, FilingResponse
)
from .client_loader import fetch_news_loader, fetch_filing_loader
from .config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Scraping Agent",
    description="Microservice for scraping news and SEC filings",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler
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
        "agent": "Scraping Agent",
        "timestamp": time.time()
    }

# News scraping endpoint
@app.get("/news", 
         response_model=NewsResponse,
         summary="Fetch news articles",
         description="Fetches news articles for a specified topic")
async def get_news(
    topic: Annotated[str, Query(description="News topic to search for")], 
    limit: Annotated[int, Query(description="Maximum number of articles")] = 5
):
    """
    Fetch news articles for a specified topic.
    
    Args:
        topic: News topic to search for
        limit: Maximum number of articles to return (default: 5)
        
    Returns:
        NewsResponse containing the scraped articles
    """
    # Input validation
    if limit < 1 or limit > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Limit must be between 1 and 20"
        )
    
    # Call document loader
    return await fetch_news_loader(topic, limit)

# SEC filing scraping endpoint
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
    # Call document loader
    return await fetch_filing_loader(str(request.filing_url))

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
