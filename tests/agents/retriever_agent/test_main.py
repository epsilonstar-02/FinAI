import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from agents.retriever_agent.main import app
from agents.retriever_agent.models import Document, SearchResult

client = TestClient(app)

@patch('agents.retriever_agent.store.vector_store')
def test_health_endpoint(mock_store):
    """Test the health check endpoint."""
    # Setup mock
    mock_store.get_stats.return_value = {"document_count": 5, "dimension": 384}
    
    # Make request
    response = client.get("/health")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["vector_store_size"] == 5

@patch('agents.retriever_agent.store.vector_store')
def test_ingest_endpoint(mock_store):
    """Test the document ingestion endpoint."""
    # Test data
    test_data = {
        "documents": [
            {"page_content": "test1", "metadata": {"source": "test"}},
            {"page_content": "test2", "metadata": {"source": "test"}}
        ]
    }
    
    # Make request
    response = client.post("/ingest", json=test_data)
    
    # Assertions
    assert response.status_code == 201
    assert response.json()["status"] == "success"
    assert response.json()["ingested"] == 2

@patch('agents.retriever_agent.store.vector_store')
def test_retrieve_endpoint(mock_store):
    """Test the document retrieval endpoint."""
    # Setup mock
    mock_result = SearchResult(
        document=Document(
            page_content="test content",
            metadata={"source": "test"}
        ),
        score=0.9
    )
    mock_store.similar_search.return_value = [mock_result]
    
    # Test data
    test_data = {"query": "test query", "top_k": 3}
    
    # Make request
    response = client.post("/retrieve", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["document"]["page_content"] == "test content"
    assert results[0]["score"] == 0.9

@patch('agents.retriever_agent.store.vector_store')
def test_retrieve_with_filter(mock_store):
    """Test retrieval with metadata filter."""
    # Setup mock
    mock_result = SearchResult(
        document=Document(
            page_content="filtered content",
            metadata={"source": "specific_source"}
        ),
        score=0.95
    )
    mock_store.similar_search.return_value = [mock_result]
    
    # Test data with filter
    test_data = {
        "query": "test query",
        "filter": {"source": "specific_source"}
    }
    
    # Make request
    response = client.post("/retrieve", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["document"]["metadata"]["source"] == "specific_source"
