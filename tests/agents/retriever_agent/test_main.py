import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agents.retriever_agent.models import Document, SearchResult

@patch('agents.retriever_agent.main.vector_store')
def test_health_endpoint(mock_store, test_client):
    """Test the health check endpoint."""
    # Setup mock
    mock_store.get_stats.return_value = {"document_count": 5, "dimension": 384}
    
    # Make request
    response = test_client.get("/health")
    
    # Assertions
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "healthy"
    assert response_data["vector_store_size"] == 5

@patch('agents.retriever_agent.main.vector_store')
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

@patch('agents.retriever_agent.main.vector_store')
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
    response_data = response.json()
    assert "results" in response_data
    results = response_data["results"]
    assert len(results) == 1
    assert results[0]["document"]["page_content"] == "test content"
    assert results[0]["score"] == 0.9

@patch('agents.retriever_agent.main.vector_store')
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
    response_data = response.json()
    assert "results" in response_data
    results = response_data["results"]
    assert len(results) == 1
    assert results[0]["document"]["metadata"]["source"] == "test"


@patch('agents.retriever_agent.main.vector_store')
def test_delete_documents_endpoint(mock_store, test_client):
    """Test the document deletion endpoint."""
    # Setup mock
    mock_store.delete_documents.return_value = 2
    
    # Make request with document_ids as query parameters
    response = test_client.delete(
        "/documents",
        params={"document_ids": ["doc1", "doc2"]}
    )
    
    # Assertions
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["deleted"] == 2
    
    # Check that delete_documents was called with the correct arguments
    mock_store.delete_documents.assert_called_once()
    args, kwargs = mock_store.delete_documents.call_args
    assert args[0] == ["doc1", "doc2"]  # First positional argument
    assert kwargs.get("namespace") is None  # namespace is passed as keyword arg


@patch('agents.retriever_agent.main.vector_store')
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


@patch('agents.retriever_agent.main.vector_store')
def test_update_document_not_found(mock_store, test_client):
    """Test updating a non-existent document returns appropriate error response.
    
    Verifies that:
    - API returns 404 status code for non-existent documents
    - Error response contains the expected structure with detail and message
    - Error message includes the non-existent document ID
    """
    # Setup mock to simulate document not found
    mock_store.update_document.return_value = False
    
    # Test data with a sample document
    test_data = {
        "document": {
            "page_content": "new content",
            "metadata": {"source": "test"}
        }
    }
    
    # Make request with a non-existent document ID
    document_id = "nonexistent"
    response = test_client.put(f"/documents/{document_id}", json=test_data)
    
    # Verify response status code
    assert response.status_code == 404, "Should return 404 for non-existent document"
    
    # Parse and validate response structure
    response_data = response.json()
    assert isinstance(response_data, dict), "Response should be a JSON object"
    
    # Verify error response structure
    assert "detail" in response_data, "Response should contain 'detail' field"
    assert isinstance(response_data["detail"], dict), "Detail should be an object"
    
    # Verify error message content
    error_detail = response_data["detail"]
    assert error_detail.get("detail") == "Document not found", \
           "Error detail should indicate document not found"
    assert document_id in error_detail.get("message", ""), \
           f"Error message should include document ID '{document_id}'"


@patch('agents.retriever_agent.main.vector_store')
def test_clear_vector_store_endpoint(mock_store, test_client):
    """Test the endpoint to clear the vector store."""
    # Setup mock
    mock_store.clear_vector_store.return_value = True
    
    # Make request
    response = test_client.delete("/clear")
    
    # Assertions
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["cleared"] is True
    
    # Verify the mock was called
    mock_store.clear_vector_store.assert_called_once()


@patch('agents.retriever_agent.main.vector_store')
def test_batch_ingest_endpoint(mock_store, test_client):
    """Test the batch document ingestion endpoint."""
    # Setup mock
    mock_store.add_documents_batched.return_value = ["doc1", "doc2", "doc3"]
    
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
    # Verify add_documents_batched was called with the right parameters
    mock_store.add_documents_batched.assert_called_once()
    args, kwargs = mock_store.add_documents_batched.call_args
    documents = args[0]
    assert len(documents) == 3
    assert kwargs["batch_size"] == 100  # Default batch size
    # assert len(args[0]) == 5  # 5 documents
    # assert kwargs["batch_size"] == 2


@patch('agents.retriever_agent.main.vector_store')
def test_rate_limiter(mock_store, test_client):
    """Test that rate limiting works."""
    # Setup mock
    mock_store.get_stats.return_value = {"document_count": 5, "dimension": 384}
    
    # Make multiple requests in quick succession
    # Use a real IP address to ensure rate limiting is active
    test_ip = "192.168.1.1"
    headers = {"X-Forwarded-For": test_ip}
    
    # First make sure rate limit data is cleared for this IP
    from agents.retriever_agent.main import rate_limit_data
    if test_ip in rate_limit_data:
        del rate_limit_data[test_ip]
    
    # First 100 requests should succeed
    for i in range(100):
        response = test_client.get("/health", headers=headers)
        assert response.status_code == 200, f"Request {i+1} failed with status {response.status_code}"
    
    # The 101st request should be rate limited
    response = test_client.get("/health", headers=headers)
    assert response.status_code == 429, f"Expected 429 status code, got {response.status_code}"
    
    # Clean up
    if test_ip in rate_limit_data:
        del rate_limit_data[test_ip]
    assert "Rate limit exceeded" in response.json()["detail"]
    
    # Reset rate limiting for other tests
    from agents.retriever_agent.main import rate_limit_data
    if test_ip in rate_limit_data:
        del rate_limit_data[test_ip]
