from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Document(BaseModel):
    """A document with content and optional metadata."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    """Request model for ingesting documents."""
    documents: List[Document]
    namespace: Optional[str] = None

class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str
    top_k: Optional[int] = None
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    """A single search result with document and score."""
    document: Document
    score: float

class QueryResponse(BaseModel):
    """Response model for document queries."""
    results: List[SearchResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    vector_store_size: int = 0
