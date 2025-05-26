from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from .models import (
    IngestRequest,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    Document as DocModel
)
from .store import vector_store
from .config import settings

app = FastAPI(
    title="Retriever Agent",
    description="Semantic search and document retrieval service",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "model": settings.EMBEDDING_MODEL,
        "vector_store_size": stats.get("document_count", 0)
    }

@app.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store.
    """
    try:
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
        
        return {"status": "success", "ingested": len(documents)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}"
        )

@app.post("/retrieve", response_model=QueryResponse)
async def retrieve_documents(request: QueryRequest):
    """
    Retrieve documents similar to the query.
    """
    try:
        results = vector_store.similar_search(
            query=request.query,
            k=request.top_k or settings.DEFAULT_TOP_K,
            filter=request.filter,
            namespace=request.namespace
        )
        
        return QueryResponse(
            results=results,
            metadata={
                "query": request.query,
                "top_k": request.top_k or settings.DEFAULT_TOP_K,
                "namespace": request.namespace or "default"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

if __name__ == "__main__":
    # For local development
    uvicorn.run("agents.retriever_agent.main:app", host="0.0.0.0", port=8003, reload=True)
