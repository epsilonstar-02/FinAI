import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
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


@patch('agents.retriever_agent.store.vector_store')
def test_delete_documents_endpoint(mock_store):
    """Test the document deletion endpoint."""
    # Setup mock
    mock_store.delete_documents.return_value = 2
    
    # Test data
    test_data = {"document_ids": ["doc1", "doc2"]}
    
    # Make request
    response = client.delete("/documents", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["deleted"] == 2
    mock_store.delete_documents.assert_called_once_with(
        ["doc1", "doc2"], namespace=None
    )


@patch('agents.retriever_agent.store.vector_store')
def test_update_document_endpoint(mock_store):
    """Test the document update endpoint."""
    # Setup mock
    mock_store.update_document.return_value = True
    
    # Test data
    test_data = {
        "document_id": "doc1",
        "document": {
            "page_content": "updated content", 
            "metadata": {"source": "updated_source"}
        }
    }
    
    # Make request
    response = client.put("/documents/doc1", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["updated"] is True


@patch('agents.retriever_agent.store.vector_store')
def test_update_document_not_found(mock_store):
    """Test updating a document that doesn't exist."""
    # Setup mock
    mock_store.update_document.return_value = False
    
    # Test data
    test_data = {
        "document_id": "non_existent",
        "document": {
            "page_content": "updated content", 
            "metadata": {"source": "updated_source"}
        }
    }
    
    # Make request
    response = client.put("/documents/non_existent", json=test_data)
    
    # Assertions
    assert response.status_code == 404
    assert response.json()["status"] == "error"
    assert "not found" in response.json()["message"]


@patch('agents.retriever_agent.store.vector_store')
def test_clear_vector_store_endpoint(mock_store):
    """Test the endpoint to clear the vector store."""
    # Make request
    response = client.delete("/clear")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "cleared" in response.json()["message"]
    mock_store.clear_vector_store.assert_called_once()


@patch('agents.retriever_agent.store.vector_store')
def test_batch_ingest_endpoint(mock_store):
    """Test the batch document ingestion endpoint."""
    # Setup mock
    mock_store.add_documents_batched.return_value = 5
    
    # Test data with multiple documents
    test_data = {
        "documents": [
            {"page_content": f"test{i}", "metadata": {"batch": "test"}} 
            for i in range(5)
        ],
        "batch_size": 2
    }
    
    # Make request
    response = client.post("/batch_ingest", json=test_data)
    
    # Assertions
    assert response.status_code == 201
    assert response.json()["status"] == "success"
    assert response.json()["ingested"] == 5
    # Verify add_documents_batched was called with the right parameters
    mock_store.add_documents_batched.assert_called_once()
    args, kwargs = mock_store.add_documents_batched.call_args
    assert len(args[0]) == 5  # 5 documents
    assert kwargs["batch_size"] == 2


@patch('agents.retriever_agent.store.vector_store')
def test_rate_limiter(mock_store):
    """Test that rate limiting works."""
    # Make multiple rapid requests to trigger rate limiting
    responses = []
    for _ in range(10):
        responses.append(client.get("/health"))
    
    # At least some of the responses should have a 429 status code
    status_codes = [r.status_code for r in responses]
    
    # Since rate limiting is configured at the app level and may depend on test execution environment,
    # we just check that rate limiting behavior exists rather than exact counts
    assert 429 in status_codes or len(set(status_codes)) > 1
