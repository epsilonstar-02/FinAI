import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agents.retriever_agent.models import Document, SearchResult

@patch('agents.retriever_agent.store.vector_store')
def test_health_endpoint(mock_store, test_client):
    """Test the health check endpoint."""
    # Setup mock
    mock_store.get_stats.return_value = {"document_count": 5, "dimension": 384}
    
    # Make request
    response = test_client.get("/health")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["vector_store_size"] == 5

@patch('agents.retriever_agent.store.vector_store')
def test_ingest_endpoint(mock_store, test_client):
    """Test the document ingestion endpoint."""
    # Test data
    test_data = {
        "documents": [
            {"page_content": "test1", "metadata": {"source": "test"}},
            {"page_content": "test2", "metadata": {"source": "test"}}
        ]
    }
    
    # Make request
    response = test_client.post("/ingest", json=test_data)
    
    # Assertions
    assert response.status_code == 201
    assert response.json()["status"] == "success"
    assert response.json()["ingested"] == 2

@patch('agents.retriever_agent.store.vector_store')
def test_retrieve_endpoint(mock_store, test_client):
    """Test the document retrieval endpoint."""
    # Setup mock
    mock_result = SearchResult(
        document=Document(
            page_content="test content",
            metadata={"source": "test"}
        ),
        score=0.9
    )
    mock_store.similarity_search.return_value = [mock_result]
    
    # Test data
    test_data = {"query": "test query", "top_k": 1}
    
    # Make request
    response = test_client.post("/retrieve", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["document"]["page_content"] == "test content"
    assert results[0]["score"] == 0.9

@patch('agents.retriever_agent.store.vector_store')
def test_retrieve_with_filter(mock_store, test_client):
    """Test retrieval with metadata filter."""
    # Setup mock
    mock_result = SearchResult(
        document=Document(
            page_content="test content",
            metadata={"source": "test", "type": "test_type"}
        ),
        score=0.9
    )
    mock_store.similarity_search.return_value = [mock_result]
    
    # Test data with filter
    test_data = {
        "query": "test query",
        "top_k": 3,
        "filter": {"source": "test"}
    }
    
    # Make request
    response = test_client.post("/retrieve", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["document"]["metadata"]["source"] == "test"


@patch('agents.retriever_agent.store.vector_store')
def test_delete_documents_endpoint(mock_store, test_client):
    """Test the document deletion endpoint."""
    # Setup mock
    mock_store.delete_documents.return_value = 2
    
    # Test data
    test_data = {"document_ids": ["doc1", "doc2"]}
    
    # Make request
    response = test_client.request("DELETE", "/documents", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["deleted"] == 2
    mock_store.delete_documents.assert_called_once_with(["doc1", "doc2"], namespace=None)


@patch('agents.retriever_agent.store.vector_store')
def test_update_document_endpoint(mock_store, test_client):
    """Test the document update endpoint."""
    # Setup mock
    mock_store.update_document.return_value = True
    
    # Test data
    update_data = {
        "document_id": "doc1",
        "document": {
            "page_content": "updated content",
            "metadata": {"source": "test"}
        }
    }
    
    # Make request
    response = test_client.put("/documents/doc1", json=update_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["updated"] is True


@patch('agents.retriever_agent.store.vector_store')
def test_update_document_not_found(mock_store, test_client):
    """Test updating a non-existent document."""
    # Setup mock
    mock_store.update_document.side_effect = ValueError("Document not found")
    
    # Test data
    test_data = {
        "document_id": "nonexistent",
        "document": {
            "page_content": "new content",
            "metadata": {"source": "test"}
        }
    }
    
    # Make request
    response = test_client.put("/documents/nonexistent", json=test_data)
    
    # Assertions
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]
    assert "not found" in response.json()["message"]


@patch('agents.retriever_agent.store.vector_store')
def test_clear_vector_store_endpoint(mock_store, test_client):
    """Test the endpoint to clear the vector store."""
    # Setup mock
    mock_store.clear.return_value = True
    
    # Make request
    response = test_client.delete("/clear")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["cleared"] is True
    mock_store.clear.assert_called_once()


@patch('agents.retriever_agent.store.vector_store')
def test_batch_ingest_endpoint(mock_store, test_client):
    """Test the batch document ingestion endpoint."""
    # Setup mock
    mock_store.add_documents.return_value = ["doc1", "doc2", "doc3"]
    
    # Test data
    test_data = {
        "documents": [
            {"page_content": f"doc {i}", "metadata": {"source": "batch_test"}} 
            for i in range(3)
        ]
    }
    
    # Make request
    response = test_client.post("/ingest/batch", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert len(result["document_ids"]) == 3
    assert mock_store.add_documents.call_count == 1
    # Verify add_documents_batched was called with the right parameters
    # mock_store.add_documents_batched.assert_called_once()
    # args, kwargs = mock_store.add_documents_batched.call_args
    # assert len(args[0]) == 5  # 5 documents
    # assert kwargs["batch_size"] == 2


@patch('agents.retriever_agent.store.vector_store')
def test_rate_limiter(mock_store, test_client):
    """Test that rate limiting works."""
    # Setup mock
    mock_store.get_stats.return_value = {"document_count": 5, "dimension": 384}
    
    # Make multiple requests in quick succession
    for i in range(105):  # Just over the limit of 100 requests per minute
        response = test_client.get("/health")
        if i < 100:
            assert response.status_code == 200
        else:
            # The 101st request should be rate limited
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]
