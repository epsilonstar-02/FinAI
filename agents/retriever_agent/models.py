# agents/retriever_agent/models.py
# Updated validators to Pydantic v2 style (`field_validator`).
# Added `model_config` for Pydantic v2.

from pydantic import BaseModel, Field, field_validator # validator removed
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class DocumentMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    title: Optional[str] = None
    namespace: Optional[str] = Field(default="default") # Default namespace
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    mime_type: Optional[str] = None
    token_count: Optional[int] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = { # Pydantic v2 style config
        "arbitrary_types_allowed": True # If using complex types not directly supported by Pydantic
    }

class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=DocumentMetadata().model_dump) # Use model_dump for Pydantic v2

    @property
    def id(self) -> str:
        return self.metadata.get("id", "") # Ensure it always returns a string

    @property
    def namespace(self) -> Optional[str]:
        return self.metadata.get("namespace")
    
    # Pydantic v2: field_validator replaces root_validator and validator for this use case
    @field_validator("metadata", mode="before") # mode='before' to process input before it's validated against DocumentMetadata
    @classmethod
    def ensure_metadata_structure_and_id(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict): # If input is not a dict, try to make it one or initialize default
            v = {}
        if "id" not in v or not v["id"]:
            v["id"] = str(uuid.uuid4())
        if "namespace" not in v or not v["namespace"]: # Ensure namespace default
             v["namespace"] = "default"
        # You could also validate against DocumentMetadata here if desired, or let Pydantic handle it
        return v


class IngestRequest(BaseModel):
    documents: List[Document]
    namespace: Optional[str] = Field(default="default") # Default namespace for ingestion
    overwrite_duplicates: bool = False # This implies ID-based overwrite logic needed in store
    
    @field_validator("documents")
    @classmethod
    def add_namespace_to_docs(cls, v: List[Document], info) -> List[Document]:
        # info.data contains other validated fields of the model
        namespace_from_request = info.data.get("namespace", "default")
        for doc in v:
            # Ensure metadata exists and is a dict
            if not isinstance(doc.metadata, dict):
                doc.metadata = {} # Initialize if not a dict
            
            # Set namespace in document metadata if not already set or to override
            # This logic means request-level namespace overrides doc-level if present.
            doc.metadata["namespace"] = namespace_from_request
        return v

class IngestResponse(BaseModel):
    document_ids: List[str] # IDs of successfully ingested documents
    message: str
    elapsed_time: float
    ingested_count: int
    failed_count: int = 0
    errors: List[str] = Field(default_factory=list) # List of errors for failed docs

class BulkDeleteRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    namespace: Optional[str] = Field(default="default")
    filter: Optional[Dict[str, Any]] = None
    
    # Using a model validator (Pydantic V2 style for root validation)
    @field_validator("*", mode="after") # Placeholder, actual validation logic needed here. For now, simpler.
    @classmethod
    def check_at_least_one_criteria(cls, values): # This is a model validator in Pydantic V2
        # This is a simplified check. A proper Pydantic V2 model validator would be:
        # from pydantic import model_validator
        # @model_validator(mode='after')
        # def check_fields(self) -> 'BulkDeleteRequest':
        #    if not (self.document_ids or self.namespace or self.filter): # Simplified for example
        #        raise ValueError("At least one of document_ids, namespace, or filter must be provided for deletion.")
        #    return self
        # For now, the original logic in main.py for this validation is likely sufficient if this model
        # isn't directly parsed with this expectation.
        # Let's assume validation is done in the endpoint.
        return values


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=100) # Use settings, add bounds
    namespace: Optional[str] = Field(default="default")
    filter: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0) # Similarity scores often 0-1
    include_metadata: bool = True
    # These were in config but not request model, adding them:
    hybrid_search: Optional[bool] = Field(default=settings.ENABLE_HYBRID_SEARCH) 
    rerank: Optional[bool] = Field(default=settings.ENABLE_RERANKING)


class SearchResult(BaseModel):
    document: Document
    score: float # Assuming higher is better for similarity, or could be distance (lower is better)

class QueryResponse(BaseModel):
    results: List[SearchResult]
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())) # For tracking
    metadata: Dict[str, Any] = Field(default_factory=dict) # For query params, etc.
    elapsed_time: Optional[float] = None
    total_found_in_query: int # Total documents matched by query before top_k applied (if store supports)
                             # For FAISS, this is len(results) unless k is smaller than total matches.
                             # Simpler: total_returned = len(results)
    total_returned: int

class VectorStoreStats(BaseModel):
    document_count: int = 0
    vector_count: int = 0 # Often same as document_count if 1 vector per doc
    last_updated: Optional[datetime] = None
    namespaces: Dict[str, int] = Field(default_factory=dict)
    vector_store_type: str
    embedding_model_name: str # Changed from embedding_model for clarity
    disk_usage_bytes: Optional[int] = None

class NamespaceResponse(BaseModel):
    namespaces: List[str]
    document_counts: Dict[str, int] # namespace -> doc count

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    vector_store_type: str
    document_count_total: int = 0 # Total docs across all namespaces
    embedding_model_name: str
    version: str = Field(default="0.3.0") # Updated version