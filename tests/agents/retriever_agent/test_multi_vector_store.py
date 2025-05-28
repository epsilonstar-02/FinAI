"""
Unit tests for the multi_vector_store module of the Retriever Agent.
"""
import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
from pathlib import Path
import asyncio
import json

from fastapi.testclient import TestClient

# Add parent directory to path to import the agent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from agents.retriever_agent.multi_vector_store import (
    MultiVectorStore, 
    VectorStoreType, 
    EmbeddingModelType,
    get_multi_vector_store
)
from agents.retriever_agent.models import Document, SearchResult
from agents.retriever_agent.config import settings


class TestMultiVectorStore(unittest.TestCase):
    """Test cases for the MultiVectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the vector store
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vector_store_path = self.temp_dir.name
        
        # Create a test document
        self.test_doc = Document(
            page_content="This is a test document about finance and investments.",
            metadata={"source": "test", "id": "test-doc-1"}
        )
        
        # Mock the embedding model
        self.embedding_mock = MagicMock()
        self.embedding_mock.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        self.embedding_mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Patch the _init_embedding_model method to return our mock
        with patch('agents.retriever_agent.multi_vector_store.MultiVectorStore._init_embedding_model', 
                  return_value=self.embedding_mock):
            # Patch FAISS to avoid actual index creation
            with patch('agents.retriever_agent.multi_vector_store.FAISS') as self.faiss_mock:
                # Setup the mocked FAISS vector store
                self.mock_vs = MagicMock()
                self.faiss_mock.from_documents.return_value = self.mock_vs
                self.faiss_mock.load_local.return_value = self.mock_vs
                
                # Initialize the multi vector store with FAISS (free and open-source)
                self.store = MultiVectorStore(
                    vector_store_type=VectorStoreType.FAISS,
                    embedding_model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
                    embedding_model_name="all-MiniLM-L6-v2",
                    persist_directory=self.vector_store_path,
                    collection_name="test_collection"
                )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test that the vector store initializes correctly."""
        self.assertEqual(self.store.vector_store_type, VectorStoreType.FAISS)
        self.assertEqual(self.store.embedding_model_type, EmbeddingModelType.SENTENCE_TRANSFORMERS)
        self.assertEqual(self.store.embedding_model_name, "all-MiniLM-L6-v2")
        self.assertEqual(self.store.persist_directory, Path(self.vector_store_path))
        self.assertEqual(self.store.collection_name, "test_collection")
        self.assertEqual(self.store.document_count, 0)
    
    def test_add_documents(self):
        """Test adding documents to the vector store."""
        # Setup mock for add_documents
        self.store.vector_store = MagicMock()
        
        # Add the test document
        result = self.store.add_documents([self.test_doc])
        
        # Verify that add_documents was called on the vector store
        self.store.vector_store.add_documents.assert_called_once()
        
        # Verify that the document count was updated
        self.assertEqual(self.store.document_count, 1)
        
        # Verify that the document ID was returned
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], self.test_doc.metadata["id"])
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Setup mock for similarity_search_with_score
        self.store.vector_store = MagicMock()
        
        # Create a mock document and score
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"id": "test-id"}
        
        # Setup the return value for similarity_search_with_score
        self.store.vector_store.similarity_search_with_score.return_value = [
            (mock_doc, 0.8)
        ]
        
        # Perform similarity search
        results = self.store.similarity_search("test query", k=1)
        
        # Verify that similarity_search_with_score was called correctly
        self.store.vector_store.similarity_search_with_score.assert_called_with(
            query="test query",
            k=1,
            filter=None
        )
        
        # Verify the results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].document.page_content, "Test content")
        self.assertEqual(results[0].document.metadata["id"], "test-id")
        self.assertEqual(results[0].score, 0.8)
    
    def test_delete_documents(self):
        """Test deleting documents from the vector store."""
        # For FAISS, we need to mock the index_to_docstore
        mock_docstore = {
            "1": MagicMock(metadata={"id": "test-doc-1"}),
            "2": MagicMock(metadata={"id": "test-doc-2"})
        }
        
        self.store.vector_store = MagicMock()
        self.store.vector_store.index_to_docstore = mock_docstore
        
        # Add to document cache
        self.store.document_cache = {"test-doc-1": self.test_doc}
        self.store.document_count = 2
        
        # Setup for FAISS from_documents (used in delete for FAISS)
        self.faiss_mock.from_documents.return_value = self.mock_vs
        
        # Delete one document
        result = self.store.delete_documents(["test-doc-1"])
        
        # Verify FAISS.from_documents was called (FAISS rebuild approach)
        self.faiss_mock.from_documents.assert_called_once()
        
        # Verify the document count was updated
        self.assertEqual(self.store.document_count, 1)
        
        # Verify the document was removed from cache
        self.assertNotIn("test-doc-1", self.store.document_cache)
        
        # Verify the correct number of documents was deleted
        self.assertEqual(result, 1)
    
    def test_get_stats(self):
        """Test retrieving vector store statistics."""
        stats = self.store.get_stats()
        
        self.assertEqual(stats["document_count"], 0)
        self.assertEqual(stats["vector_store_type"], VectorStoreType.FAISS)
        self.assertEqual(stats["embedding_model_type"], EmbeddingModelType.SENTENCE_TRANSFORMERS)
        self.assertEqual(stats["embedding_model_name"], "all-MiniLM-L6-v2")
    
    def test_singleton_pattern(self):
        """Test that get_multi_vector_store returns a singleton instance."""
        with patch('agents.retriever_agent.multi_vector_store.MultiVectorStore') as mock_mvs:
            # Reset the singleton first
            import agents.retriever_agent.multi_vector_store
            agents.retriever_agent.multi_vector_store._multi_vector_store = None
            
            # First call should create a new instance
            first = get_multi_vector_store()
            mock_mvs.assert_called_once()
            
            # Reset the mock to check if it's called again
            mock_mvs.reset_mock()
            
            # Second call should return the existing instance
            second = get_multi_vector_store()
            mock_mvs.assert_not_called()
            
            # Both calls should return the same instance
            self.assertEqual(first, second)


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
