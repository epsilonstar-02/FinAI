import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from agents.retriever_agent.store import VectorStore
from agents.retriever_agent.models import Document

@patch('faiss.IndexFlatL2')
def test_vector_store_init(mock_faiss):
    """Test vector store initialization."""
    store = VectorStore()
    assert store.index is None
    assert store.documents == {}
    assert store.dimension == 384  # Default for all-MiniLM-L6-v2

@patch('faiss.IndexFlatL2')
@patch('builtins.open')
@patch('json.dump')
@patch('os.makedirs')
def test_save(mock_makedirs, mock_json_dump, mock_open, mock_faiss, temp_dir):
    """Test saving the vector store."""
    store = VectorStore()
    store.index = MagicMock()
    store.documents = {"1": {"page_content": "test", "metadata": {}}}
    
    with patch('agents.retriever_agent.store.settings') as mock_settings:
        mock_settings.VECTOR_STORE_PATH = temp_dir
        store.save("test_namespace")
    
    # Verify FAISS index was saved
    store.index.save.assert_called_once()
    # Verify documents were saved
    assert mock_json_dump.called

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
def test_similar_search(mock_embedder, mock_load, temp_dir):
    """Test similarity search."""
    # Setup mocks
    mock_embedder.embed_query.return_value = [0.1] * 384
    
    store = VectorStore()
    store.index = MagicMock()
    store.index.search.return_value = ([[0.1, 0.2]], [[0, 1]])
    store.documents = {
        "0": {"page_content": "test1", "metadata": {"source": "test"}},
        "1": {"page_content": "test2", "metadata": {"source": "test"}}
    }
    
    results = store.similar_search("test query", k=2)
    
    assert len(results) == 2
    assert results[0].document.page_content in ["test1", "test2"]
    assert results[0].score in [0.1, 0.2]
