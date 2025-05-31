# agents/retriever_agent/main.py

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from .models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, SearchResult, # Added SearchResult
    HealthResponse,
    Document as AppDocument, # Renamed to avoid conflict with Langchain's Document
    NamespaceResponse,
    VectorStoreStats,
    BulkDeleteRequest # Added for the delete endpoint
)
from .multi_vector_store import get_multi_vector_store, MultiVectorStore
from .config import settings

logger = logging.getLogger(__name__)

# Get vector store instance (singleton)
def get_store() -> MultiVectorStore:
    return get_multi_vector_store()

app = FastAPI(
    title="Retriever Agent",
    description="Semantic search and document retrieval service for FinAI",
    version="0.3.0", # Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Rate limiting (simple in-memory)
_rate_limit_data: Dict[str, List[float]] = {}
_last_cleanup_time = time.time()

@app.middleware("http")
async def rate_limit_and_time_middleware(request: Request, call_next):
    global _last_cleanup_time
    client_ip = request.client.host if request.client else "unknown_ip"
    current_time = time.time()

    if current_time - _last_cleanup_time > settings.CACHE_TTL_SECONDS: # Reuse TTL for cleanup interval
        for ip in list(_rate_limit_data.keys()):
            _rate_limit_data[ip] = [ts for ts in _rate_limit_data.get(ip, []) if current_time - ts < 60] # 60s window
            if not _rate_limit_data.get(ip):
                _rate_limit_data.pop(ip, None)
        _last_cleanup_time = current_time
    
    ip_timestamps = _rate_limit_data.get(client_ip, [])
    ip_timestamps = [ts for ts in ip_timestamps if current_time - ts < 60] # Filter old
    
    if len(ip_timestamps) >= 100: # Max 100 requests per minute
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            content={"detail": "Rate limit exceeded. Please try again later."})
    
    ip_timestamps.append(current_time)
    _rate_limit_data[client_ip] = ip_timestamps
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Seconds"] = str(round(process_time, 3))
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "message": str(exc.detail)}, # Add message field
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred.", "message": str(exc)}
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(store: MultiVectorStore = Depends(get_store)):
    stats = store.get_stats() # Aggregated stats
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        vector_store_type=stats["vector_store_type"],
        document_count_total=stats["document_count"],
        embedding_model_name=stats["embedding_model_name"],
        version=app.version
    )

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def ingest_documents_endpoint(
    request: IngestRequest,
    store: MultiVectorStore = Depends(get_store)
):
    if not request.documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents provided for ingestion.")
    
    start_time = time.time()
    # Documents in request are already AppDocument type due to Pydantic validation
    # Namespace is also correctly handled by IngestRequest model's validator
    
    added_ids, failed_ids = store.add_documents(request.documents, namespace=request.namespace or "default")
    elapsed_time = time.time() - start_time
    
    if failed_ids:
        msg = f"Partially ingested {len(added_ids)} documents. Failed to ingest {len(failed_ids)} documents."
        logger.warning(msg + f" into namespace '{request.namespace}'")
        # Could return 207 Multi-Status if desired, but for now, success if any were added.
    else:
        msg = f"Successfully ingested {len(added_ids)} documents."
        logger.info(msg + f" into namespace '{request.namespace}'")

    return IngestResponse(
        document_ids=added_ids,
        message=msg,
        elapsed_time=round(elapsed_time, 3),
        ingested_count=len(added_ids),
        failed_count=len(failed_ids)
        # errors field could be populated if add_documents returned specific error messages
    )


@app.post("/retrieve", response_model=QueryResponse, tags=["Documents"])
async def retrieve_documents_endpoint(
    request: QueryRequest,
    store: MultiVectorStore = Depends(get_store)
):
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")
    
    start_time = time.time()
    results: List[SearchResult] = store.similarity_search(
        query=request.query,
        k=request.top_k,
        namespace=request.namespace or "default",
        filter_dict=request.filter,
        min_score=request.min_score
        # Hybrid search and reranking are more complex and depend on store capabilities.
        # The MultiVectorStore would need to implement logic for these based on ENABLE_ flags.
        # For now, these flags in QueryRequest are noted but not fully wired up here.
    )
    elapsed_time = time.time() - start_time
    
    return QueryResponse(
        results=results,
        query_id=str(uuid.uuid4()), # Generate a unique ID for this query response
        metadata={
            "query": request.query, "top_k_requested": request.top_k, 
            "namespace_searched": request.namespace or "default",
            "filter_applied": request.filter,
            "min_score_applied": request.min_score
        },
        elapsed_time=round(elapsed_time, 3),
        total_returned=len(results),
        total_found_in_query=len(results) # Simplified for FAISS; other stores might provide true total.
    )

@app.delete("/documents", status_code=status.HTTP_200_OK, tags=["Documents"])
async def delete_documents_endpoint(
    request: BulkDeleteRequest, # Use the model for the request body
    store: MultiVectorStore = Depends(get_store)
):
    # Validation from model is basic; more complex in endpoint
    if not request.document_ids and not request.filter and not request.namespace: # Re-check here
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                             detail="At least one of document_ids, namespace (for full clear), or filter must be provided.")

    # TODO: Implement filter-based deletion if store supports it.
    # Langchain stores' delete methods vary. Chroma supports `delete(where=...)`.
    if request.filter:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="Filter-based deletion is not yet implemented for all backends.")

    deleted_count = 0
    if request.document_ids:
        deleted_count = store.delete_documents(request.document_ids, namespace=request.namespace or "default")
    elif request.namespace and not request.document_ids and not request.filter: # Clear whole namespace
        store.clear_vector_store(namespace=request.namespace)
        # clear_vector_store doesn't return count, we assume all docs in that namespace were deleted.
        # We'd need to get count before clearing for an accurate number.
        stats_before_clear = store.get_stats(namespace=request.namespace)
        deleted_count = stats_before_clear.get("document_count", 0) # Approximate
        logger.info(f"Cleared namespace '{request.namespace}'. {deleted_count} documents (approx) removed.")
    
    return {"status": "success", "deleted_count": deleted_count, "namespace": request.namespace or "default"}


@app.put("/documents/{document_id}", status_code=status.HTTP_200_OK, tags=["Documents"])
async def update_document_endpoint(
    document_id: str,
    document_payload: AppDocument, # The document model itself, not nested in a request model
    namespace: Optional[str] = Query("default", description="Namespace of the document"),
    store: MultiVectorStore = Depends(get_store)
):
    if not document_payload.page_content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document content cannot be empty.")

    # Ensure the payload's metadata ID matches path param if metadata is provided, or set it
    if not document_payload.metadata:
        document_payload.metadata = {}
    document_payload.metadata["id"] = document_id
    document_payload.metadata["namespace"] = namespace or "default"
    
    success = store.update_document(document_id, document_payload, namespace=namespace or "default")
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Document with ID '{document_id}' not found in namespace '{namespace}'.")
    return {"status": "success", "updated_id": document_id, "namespace": namespace or "default"}


@app.get("/stats", response_model=VectorStoreStats, tags=["System"])
async def get_store_stats(
    namespace: Optional[str] = Query(None, description="Optional: Get stats for a specific namespace. If None, returns aggregated stats."),
    store: MultiVectorStore = Depends(get_store)
):
    stats_data = store.get_stats(namespace=namespace) # get_stats already returns a dict matching VectorStoreStats
    return VectorStoreStats(**stats_data)


@app.get("/namespaces", response_model=NamespaceResponse, tags=["System"])
async def list_namespaces_endpoint(store: MultiVectorStore = Depends(get_store)):
    namespaces_list = store.list_namespaces()
    counts = {ns: store.get_stats(ns).get("document_count",0) for ns in namespaces_list}
    return NamespaceResponse(namespaces=namespaces_list, document_counts=counts)


# Removed /ingest/batch as /ingest already takes a list of documents.
# If the intent of /ingest/batch was to control how the *server* internally batches
# calls to store.add_documents, that logic would go here, iterating over request.documents.
# But store.add_documents(List[AppDocument]) itself is a batch operation.

if __name__ == "__main__":
    import uvicorn
    # Corrected uvicorn run command for module structure
    uvicorn.run("agents.retriever_agent.main:app", host=settings.HOST, port=8003, reload=True, log_level=settings.LOG_LEVEL.lower())