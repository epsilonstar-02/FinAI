from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import logging
import time
import uvicorn
from datetime import datetime

from .models import (
    IngestRequest,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    Document as DocModel
)
from .store import vector_store
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting settings
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS = 100  # requests per duration
rate_limit_data = {}

app = FastAPI(
    title="Retriever Agent",
    description="Semantic search and document retrieval service for FinAI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing and rate limiting middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Rate limiting check
    client_ip = request.client.host
    current_time = time.time()
    
    # Initialize or clean up expired entries
    if client_ip in rate_limit_data:
        # Remove timestamps older than the rate limit duration
        rate_limit_data[client_ip] = [ts for ts in rate_limit_data[client_ip] 
                                   if current_time - ts < RATE_LIMIT_DURATION]
    else:
        rate_limit_data[client_ip] = []
    
    # Check if rate limit exceeded
    if len(rate_limit_data[client_ip]) >= MAX_REQUESTS:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            content={"detail": "Rate limit exceeded. Please try again later."},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )
    
    # Add current request timestamp
    rate_limit_data[client_ip].append(current_time)
    
    # Process timing
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "model": settings.EMBEDDING_MODEL,
        "vector_store_size": stats.get("document_count", 0),
        "timestamp": datetime.utcnow()
    }

@app.post("/ingest", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store.
    
    - **documents**: List of documents to ingest
    - **namespace**: Optional namespace to organize documents (default: "default")
    """
    try:
        # Validate request
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided for ingestion"
            )
            
        # Convert to internal Document model
        documents = [
            DocModel(
                page_content=doc.page_content,
                metadata=doc.metadata or {}
            )
            for doc in request.documents
        ]
        
        # Add to vector store
        vector_store.add_documents(documents, namespace=request.namespace)
        
        logger.info(f"Ingested {len(documents)} documents into namespace '{request.namespace or 'default'}'")
        
        return {
            "status": "success", 
            "ingested": len(documents),
            "namespace": request.namespace or "default",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}"
        )

@app.post("/retrieve", response_model=QueryResponse, tags=["Documents"])
async def retrieve_documents(request: QueryRequest):
    """
    Retrieve documents similar to the query.
    
    - **query**: The search query text
    - **top_k**: Number of results to return (default: 5)
    - **namespace**: Optional namespace to search in (default: "default")
    - **filter**: Optional metadata filters to apply
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
            
        # Get top_k from request or use default
        top_k = request.top_k or settings.DEFAULT_TOP_K
        
        # Cap top_k to reasonable limit
        if top_k > 100:
            logger.warning(f"Requested top_k={top_k} exceeds limit, capping to 100")
            top_k = 100
            
        # Perform similarity search    
        results = vector_store.similarity_search(
            query=request.query,
            k=top_k,
            filter=request.filter,
            namespace=request.namespace
        )
        
        process_time = time.time() - start_time
        logger.info(f"Retrieved {len(results)} documents for query in {process_time:.2f}s")
        
        return QueryResponse(
            results=results,
            metadata={
                "query": request.query,
                "top_k": top_k,
                "namespace": request.namespace or "default",
                "filter": request.filter,
                "result_count": len(results),
                "process_time_seconds": round(process_time, 3),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Document retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

# Document deletion endpoint
@app.delete("/documents", tags=["Documents"])
async def delete_documents(document_ids: List[str], namespace: Optional[str] = None):
    """
    Delete documents from the vector store by their IDs.
    
    - **document_ids**: List of document IDs to delete
    - **namespace**: Optional namespace the documents are in
    """
    try:
        if not document_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No document IDs provided for deletion"
            )
            
        deleted_count = vector_store.delete_documents(document_ids, namespace)
        
        return {
            "status": "success", 
            "deleted": deleted_count,
            "namespace": namespace or "default",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {str(e)}"
        )

# Clear vector store endpoint
@app.delete("/clear", tags=["Documents"])
async def clear_store(namespace: Optional[str] = None):
    """
    Clear all documents from the vector store for a specific namespace.
    
    - **namespace**: Optional namespace to clear (default: "default")
    """
    try:
        vector_store.clear_vector_store(namespace)
        
        return {
            "status": "success",
            "message": f"Cleared all documents from namespace: {namespace or 'default'}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Vector store clear operation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear vector store: {str(e)}"
        )

# Batch ingestion endpoint
@app.post("/batch-ingest", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def batch_ingest_documents(request: IngestRequest, batch_size: int = 100):
    """
    Ingest a large number of documents into the vector store in batches.
    
    - **documents**: List of documents to ingest
    - **namespace**: Optional namespace (default: "default")
    - **batch_size**: Size of each processing batch (default: 100)
    """
    try:
        # Validate request
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided for ingestion"
            )
            
        if batch_size <= 0 or batch_size > 500:
            batch_size = 100  # Use default if invalid
            
        # Convert to internal Document model
        documents = [
            DocModel(
                page_content=doc.page_content,
                metadata=doc.metadata or {}
            )
            for doc in request.documents
        ]
        
        # Add documents in batches
        vector_store.add_documents_batched(documents, batch_size, request.namespace)
        
        logger.info(f"Batch ingested {len(documents)} documents into namespace '{request.namespace or 'default'}'")
        
        return {
            "status": "success", 
            "ingested": len(documents),
            "namespace": request.namespace or "default",
            "batches": (len(documents) - 1) // batch_size + 1,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch document ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch ingest documents: {str(e)}"
        )

# Document update endpoint
class DocumentUpdateRequest(BaseModel):
    document_id: str
    document: DocModel
    namespace: Optional[str] = None

@app.put("/documents", tags=["Documents"])
async def update_document(request: DocumentUpdateRequest):
    """
    Update a document in the vector store.
    
    - **document_id**: ID of the document to update
    - **document**: New document content and metadata
    - **namespace**: Optional namespace (default: "default")
    """
    try:
        success = vector_store.update_document(
            document_id=request.document_id,
            document=request.document,
            namespace=request.namespace
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {request.document_id} not found"
            )
        
        return {
            "status": "success",
            "message": f"Updated document {request.document_id}",
            "namespace": request.namespace or "default",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )

# Get statistics about the vector store
@app.get("/stats", tags=["System"])
async def get_stats(namespace: Optional[str] = None):
    """
    Get statistics about the vector store.
    
    - **namespace**: Optional namespace to get stats for (default: "default")
    """
    try:
        stats = vector_store.get_stats(namespace)
        return {
            **stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

if __name__ == "__main__":
    # For local development
    logger.info("Starting Retriever Agent service")
    uvicorn.run("agents.retriever_agent.main:app", host="0.0.0.0", port=8003, reload=True)
