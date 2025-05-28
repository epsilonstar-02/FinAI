"""
Unit tests for the FastAPI endpoints of the Retriever Agent.
"""
import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import json
from fastapi.testclient import TestClient

# Add parent directory to path to import the agent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from agents.retriever_agent.main import app
from agents.retriever_agent.models import (
    Document, 
    IngestRequest, 
    QueryRequest, 
    QueryResponse,
    SearchResult,
    HealthResponse
)
from agents.retriever_agent.multi_vector_store import MultiVectorStore, get_multi_vector_store


class TestRetrieverAgentEndpoints(unittest.TestCase):
    """Test cases for the Retriever Agent API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test client
        self.client = TestClient(app)
        
        # Create mock for the multi_vector_store
        self.mock_vector_store = MagicMock(spec=MultiVectorStore)
        
        # Patch get_multi_vector_store to return our mock
        self.patcher = patch('agents.retriever_agent.main.get_multi_vector_store', 
                            return_value=self.mock_vector_store)
        self.mock_get_vector_store = self.patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        # Setup mock for get_stats
        self.mock_vector_store.get_stats.return_value = {
            "document_count": 0,
            "vector_store_type": "faiss",
            "embedding_model_type": "sentence_transformers",
            "embedding_model_name": "all-MiniLM-L6-v2",
            "last_updated": "2025-05-28T00:00:00"
        }
        
        # Make request to health endpoint
        response = self.client.get("/health")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("timestamp", data)
        self.assertEqual(data["vector_store_size"], 0)
    
    def test_ingest(self):
        """Test the document ingestion endpoint."""
        # Setup mock for add_documents
        doc_ids = ["test-doc-1"]
        self.mock_vector_store.add_documents.return_value = doc_ids
        
        # Create test document
        test_doc = {
            "page_content": "This is a test document about finance.",
            "metadata": {"source": "test", "id": "test-doc-1"}
        }
        
        # Create request payload
        payload = {
            "documents": [test_doc],
            "namespace": "test-namespace"
        }
        
        # Make request to ingest endpoint
        response = self.client.post("/ingest", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_ids"], doc_ids)
        self.assertIn("message", data)
        
        # Verify add_documents was called with correct arguments
        self.mock_vector_store.add_documents.assert_called_once()
        call_args = self.mock_vector_store.add_documents.call_args[0]
        self.assertEqual(len(call_args[0]), 1)  # One document
        self.assertEqual(call_args[0][0].page_content, "This is a test document about finance.")
        self.assertEqual(call_args[0][0].metadata["source"], "test")
        self.assertEqual(call_args[1], "test-namespace")  # Namespace
    
    def test_query(self):
        """Test the query endpoint."""
        # Create mock search results
        mock_doc = Document(
            page_content="Finance information", 
            metadata={"source": "test"}
        )
        mock_results = [SearchResult(document=mock_doc, score=0.95)]
        
        # Setup mock for similarity_search
        self.mock_vector_store.similarity_search.return_value = mock_results
        
        # Create request payload
        payload = {
            "query": "finance information",
            "top_k": 1,
            "namespace": "test-namespace"
        }
        
        # Make request to query endpoint
        response = self.client.post("/query", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["document"]["page_content"], "Finance information")
        self.assertEqual(data["results"][0]["score"], 0.95)
        
        # Verify similarity_search was called with correct arguments
        self.mock_vector_store.similarity_search.assert_called_with(
            query="finance information",
            k=1,
            filter=None,
            namespace="test-namespace"
        )
    
    def test_delete(self):
        """Test the delete endpoint."""
        # Setup mock for delete_documents
        self.mock_vector_store.delete_documents.return_value = 1
        
        # Create request payload
        payload = {
            "document_ids": ["test-doc-1"],
            "namespace": "test-namespace"
        }
        
        # Make request to delete endpoint
        response = self.client.post("/delete", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["deleted_count"], 1)
        self.assertIn("message", data)
        
        # Verify delete_documents was called with correct arguments
        self.mock_vector_store.delete_documents.assert_called_with(
            document_ids=["test-doc-1"],
            namespace="test-namespace"
        )
    
    def test_namespaces(self):
        """Test the namespaces endpoint."""
        # Setup mock for get_stats
        self.mock_vector_store.get_stats.return_value = {
            "document_count": 5,
            "namespaces": {
                "ns1": 2,
                "ns2": 3
            }
        }
        
        # Make request to namespaces endpoint
        response = self.client.get("/namespaces")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["namespaces"]), 2)
        self.assertEqual(data["document_counts"]["ns1"], 2)
        self.assertEqual(data["document_counts"]["ns2"], 3)
    
    def test_error_handling(self):
        """Test error handling in endpoints."""
        # Setup mock to raise an exception
        self.mock_vector_store.add_documents.side_effect = ValueError("Test error")
        
        # Create test document
        test_doc = {
            "page_content": "This is a test document.",
            "metadata": {"source": "test"}
        }
        
        # Create request payload
        payload = {
            "documents": [test_doc]
        }
        
        # Make request to ingest endpoint
        response = self.client.post("/ingest", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Test error", data["detail"])


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
