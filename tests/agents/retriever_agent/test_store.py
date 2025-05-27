import pytest
import os
import sys
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open as mock_open_fn
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agents.retriever_agent.store import VectorStore
from agents.retriever_agent.models import Document, SearchResult

@patch('faiss.IndexFlatL2')
def test_vector_store_init(mock_faiss):
    """Test vector store initialization."""
    store = VectorStore()
    assert store.index is None
    assert store.documents == {}
    assert store.dimension == 384  # Default for all-MiniLM-L6-v2
    assert isinstance(store.last_updated, datetime)

@patch('faiss.write_index')
@patch('builtins.open', new_callable=mock_open_fn)
@patch('json.dump')
@patch('os.makedirs')
def test_save(mock_makedirs, mock_json_dump, mock_open, mock_write_index, temp_dir):
    """Test saving the vector store."""
    store = VectorStore()
    store.index = MagicMock()
    store.documents = {"1": {"page_content": "test", "metadata": {}}}
    
    with patch('agents.retriever_agent.store.settings') as mock_settings:
        mock_settings.VECTOR_STORE_PATH = temp_dir
        store.save("test_namespace")
    
    # Verify FAISS index was saved
    assert mock_write_index.called
    # Verify documents were saved with metadata
    mock_json_dump.assert_called_once()
    # Check that the first arg to json.dump is a dict with documents and metadata
    call_args = mock_json_dump.call_args[0]
    assert "documents" in call_args[0]
    assert "metadata" in call_args[0]

@patch('faiss.read_index')
@patch('builtins.open')
@patch('json.load')
@patch('os.path.exists', return_value=True)
def test_load(mock_exists, mock_json_load, mock_open, mock_read_index, temp_dir):
    """Test loading the vector store."""
    # Setup mocks
    mock_index = MagicMock()
    mock_index.d = 384
    mock_read_index.return_value = mock_index
    mock_json_load.return_value = {"1": {"page_content": "test", "metadata": {}}}
    
    store = VectorStore()
    with patch('agents.retriever_agent.store.settings') as mock_settings:
        mock_settings.VECTOR_STORE_PATH = temp_dir
        result = store.load("test_namespace")
    
    assert result is True
    assert store.index is not None
    assert store.documents == {"1": {"page_content": "test", "metadata": {}}}

@patch('agents.retriever_agent.store.VectorStore.load')
@patch('agents.retriever_agent.store.VectorStore.save')
@patch('agents.retriever_agent.embedder.embedder')
def test_add_documents(mock_embedder, mock_save, mock_load, temp_dir):
    """Test adding documents to the vector store."""
    # Setup mocks
    mock_embedder.embed_documents.return_value = [[0.1] * 384] * 2
    
    store = VectorStore()
    store.index = MagicMock()
    
    documents = [
        Document(page_content="test1", metadata={"source": "test"}),
        Document(page_content="test2", metadata={"source": "test"})
    ]
    
    with patch('agents.retriever_agent.store.settings') as mock_settings:
        mock_settings.VECTOR_STORE_PATH = temp_dir
        store.add_documents(documents, "test_namespace")
    
    # Verify documents were added
    assert len(store.documents) == 2
    assert mock_save.called
    store.index.add.assert_called_once()

@patch('agents.retriever_agent.store.VectorStore.load')
@patch('agents.retriever_agent.embedder.embedder')
def test_similarity_search(mock_embedder, mock_load, temp_dir):
    """Test similarity search."""
    # Setup mocks
    mock_embedder.embed_query.return_value = [0.1] * 384
    mock_load.return_value = True
    
    store = VectorStore()
    store.index = MagicMock()
    store.index.search.return_value = ([[0.1, 0.2]], [[0, 1]])
    store.documents = {
        "0": {"page_content": "test1", "metadata": {"source": "test"}},
        "1": {"page_content": "test2", "metadata": {"source": "test"}}
    }
    
    results = store.similarity_search("test query", k=2)
    
    assert len(results) == 2
    assert results[0].document.page_content in ["test1", "test2"]
    assert results[0].score in [0.1, 0.2]
    
    # Test with filter
    store.documents = {
        "0": {"page_content": "test1", "metadata": {"category": "finance"}},
        "1": {"page_content": "test2", "metadata": {"category": "tech"}}
    }
    
    results = store.similarity_search("test query", k=2, filter={"category": "tech"})
    
    # Should only return the document with category=tech
    assert len(results) == 1
    assert results[0].document.page_content == "test2"


@patch('agents.retriever_agent.store.VectorStore.load')
@patch('agents.retriever_agent.store.VectorStore._rebuild_index')
def test_delete_documents(mock_rebuild_index, mock_load, temp_dir):
    """Test deleting documents from the vector store."""
    # Setup
    mock_load.return_value = True
    
    store = VectorStore()
    store.documents = {
        "doc1": {"page_content": "test1", "metadata": {}},
        "doc2": {"page_content": "test2", "metadata": {}},
        "doc3": {"page_content": "test3", "metadata": {}}
    }
    
    with patch('agents.retriever_agent.store.VectorStore.save') as mock_save:
        # Delete documents
        deleted_count = store.delete_documents(["doc1", "doc3"], "test_namespace")
        
        # Verify correct documents were deleted
        assert deleted_count == 2
        assert "doc1" not in store.documents
        assert "doc2" in store.documents
        assert "doc3" not in store.documents
        
        # Verify index was rebuilt and saved
        assert mock_rebuild_index.called
        assert mock_save.called
        
        # Test delete with non-existent document
        deleted_count = store.delete_documents(["non_existent"], "test_namespace")
        assert deleted_count == 0


@patch('faiss.IndexFlatL2')
@patch('agents.retriever_agent.store.embedder')
def test_rebuild_index(mock_embedder, mock_index_flat, temp_dir):
    """Test rebuilding the index."""
    # Setup mock index instance
    mock_index_instance = MagicMock()
    mock_index_flat.return_value = mock_index_instance
    
    # Setup mock embedder
    mock_embedder.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
    
    # Create a test store with some documents
    store = VectorStore()
    store.documents = {
        "doc1": {"page_content": "test1", "metadata": {}},
        "doc2": {"page_content": "test2", "metadata": {}}
    }
    
    # Call the method we're testing
    store._rebuild_index()
    
    # Verify the embedder was called with the correct document contents
    mock_embedder.embed_documents.assert_called_once_with(["test1", "test2"])
    
    # Verify a new index was created with the correct dimension
    mock_index_flat.assert_called_once_with(384)  # 384 is the expected dimension
    
    # Verify the embeddings were added to the index
    mock_index_instance.add.assert_called_once()
    
    # Get the actual embeddings that were added to the index
    call_args = mock_index_instance.add.call_args[0][0]
    assert call_args.shape == (2, 384)  # 2 documents, 384 dimensions


@patch('faiss.IndexFlatL2')
def test_clear_vector_store(mock_index_flat, temp_dir):
    """Test clearing the vector store."""
    store = VectorStore()
    store.documents = {"doc1": {"page_content": "test", "metadata": {}}}
    store.index = MagicMock()
    
    with patch('agents.retriever_agent.store.VectorStore.save') as mock_save:
        # Clear vector store
        store.clear_vector_store("test_namespace")
        
        # Verify documents were cleared
        assert store.documents == {}
        # Verify new index was created
        assert mock_index_flat.called
        # Verify changes were saved
        assert mock_save.called


@patch('agents.retriever_agent.store.VectorStore.add_documents')
def test_add_documents_batched(mock_add_documents):
    """Test adding documents in batches."""
    store = VectorStore()
    
    # Create test documents
    documents = [
        Document(page_content=f"doc{i}", metadata={}) 
        for i in range(25)
    ]
    
    # Add documents in batches of 10
    store.add_documents_batched(documents, batch_size=10, namespace="test")
    
    # Verify add_documents was called 3 times (25 docs in batches of 10)
    assert mock_add_documents.call_count == 3
    
    # Verify the batches
    batch1 = mock_add_documents.call_args_list[0][0][0]
    batch2 = mock_add_documents.call_args_list[1][0][0]
    batch3 = mock_add_documents.call_args_list[2][0][0]
    
    assert len(batch1) == 10
    assert len(batch2) == 10
    assert len(batch3) == 5


@patch('agents.retriever_agent.store.VectorStore.load')
@patch('agents.retriever_agent.store.VectorStore._rebuild_index')
@patch('agents.retriever_agent.store.VectorStore.save')
def test_update_document(mock_save, mock_rebuild_index, mock_load):
    """Test updating a document in the vector store."""
    # Setup
    mock_load.return_value = True
    
    store = VectorStore()
    store.documents = {
        "doc1": {"page_content": "original content", "metadata": {"category": "old"}}  
    }
    
    # Update existing document
    updated_doc = Document(page_content="new content", metadata={"category": "new"})
    success = store.update_document("doc1", updated_doc, "test_namespace")
    
    # Verify document was updated
    assert success is True
    assert store.documents["doc1"]["page_content"] == "new content"
    assert store.documents["doc1"]["metadata"]["category"] == "new"
    assert "updated_at" in store.documents["doc1"]
    
    # Verify index was rebuilt and saved
    assert mock_rebuild_index.called
    assert mock_save.called
    
    # Test update with non-existent document
    success = store.update_document("non_existent", updated_doc, "test_namespace")
    assert success is False


@patch('agents.retriever_agent.store.VectorStore.load')
def test_matches_filter(mock_load):
    """Test the filter matching functionality."""
    store = VectorStore()
    
    # Setup test metadata
    metadata = {
        "category": "finance",
        "source": "news",
        "date": "2023-05-01",
        "rating": 4.5,
        "tags": ["stocks", "market"]
    }
    
    # Test simple equality filter
    assert store._matches_filter(metadata, {"category": "finance"}) is True
    assert store._matches_filter(metadata, {"category": "tech"}) is False
    
    # Test multiple conditions (AND logic)
    assert store._matches_filter(metadata, {"category": "finance", "source": "news"}) is True
    assert store._matches_filter(metadata, {"category": "finance", "source": "blog"}) is False
    
    # Test with list values (OR logic)
    assert store._matches_filter(metadata, {"category": ["finance", "tech"]}) is True
    assert store._matches_filter(metadata, {"category": ["tech", "science"]}) is False
    
    # Test with range operators
    assert store._matches_filter(metadata, {"rating": {"$gt": 4.0}}) is True
    assert store._matches_filter(metadata, {"rating": {"$lt": 4.0}}) is False
    assert store._matches_filter(metadata, {"rating": {"$gte": 4.5}}) is True
    assert store._matches_filter(metadata, {"rating": {"$lte": 4.5}}) is True
    
    # Test with non-existent field
    assert store._matches_filter(metadata, {"non_existent": "value"}) is False
    
    # Test with empty filter
    assert store._matches_filter(metadata, {}) is True
