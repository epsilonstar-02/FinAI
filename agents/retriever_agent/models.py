from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime
from enum import Enum
import uuid

class DocumentMetadata(BaseModel):
    """Enhanced metadata for a document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    title: Optional[str] = None
    namespace: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    mime_type: Optional[str] = None
    token_count: Optional[int] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class Document(BaseModel):
    """A document with content and enhanced metadata."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Get document ID from metadata."""
        return self.metadata.get("id", str(uuid.uuid4()))
    
    @property
    def namespace(self) -> Optional[str]:
        """Get namespace from metadata."""
        return self.metadata.get("namespace")
    
    @validator("metadata")
    def ensure_id_in_metadata(cls, v):
        """Ensure document has an ID in metadata."""
        if "id" not in v:
            v["id"] = str(uuid.uuid4())
        return v

class IngestRequest(BaseModel):
    """Request model for ingesting documents."""
    documents: List[Document]
    namespace: Optional[str] = None
    overwrite_duplicates: bool = False
    
    @validator("documents")
    def add_namespace_to_docs(cls, v, values):
        """Add namespace to documents if provided."""
        namespace = values.get("namespace")
        if namespace:
            for doc in v:
                doc.metadata["namespace"] = namespace
        return v

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_ids: List[str]
    message: str
    elapsed_time: float
    failed_docs: int = 0

class BulkDeleteRequest(BaseModel):
    """Request model for bulk document deletion."""
    document_ids: Optional[List[str]] = None
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    
    @validator("document_ids", "namespace", "filter")
    def validate_deletion_parameters(cls, v, values, **kwargs):
        """Validate that at least one deletion parameter is provided."""
        field = kwargs.get("field")
        if field == "document_ids" and not v and not values.get("namespace") and not values.get("filter"):
            raise ValueError("At least one of document_ids, namespace, or filter must be provided")
        return v

class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str
    top_k: Optional[int] = None
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = None
    include_metadata: bool = True
    hybrid_search: Optional[bool] = None
    rerank: Optional[bool] = None

class SearchResult(BaseModel):
    """A single search result with document and score."""
    document: Document
    score: float

class QueryResponse(BaseModel):
    """Response model for document queries."""
    results: List[SearchResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    elapsed_time: Optional[float] = None
    total_found: int = 0

class VectorStoreStats(BaseModel):
    """Statistics about the vector store."""
    document_count: int = 0
    vector_count: int = 0
    last_updated: Optional[datetime] = None
    namespaces: Dict[str, int] = Field(default_factory=dict)  # namespace -> doc count
    vector_store_type: str
    embedding_model: str
    disk_usage_bytes: Optional[int] = None

class NamespaceResponse(BaseModel):
    """Response model for listing namespaces."""
    namespaces: List[str]
    document_counts: Dict[str, int]

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    vector_store_type: str
    vector_store_size: int = 0
    document_count: int = 0
    embedding_model: str
    version: str = "0.2.0"
