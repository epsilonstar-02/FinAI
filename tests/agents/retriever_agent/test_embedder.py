import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from agents.retriever_agent.embedder import Embedder

def test_embedder_singleton():
    """Test that only one instance of Embedder exists."""
    embedder1 = Embedder()
    embedder2 = Embedder()
    assert embedder1 is embedder2

@patch('sentence_transformers.SentenceTransformer')
def test_embed_documents(mock_model_class):
    """Test embedding multiple documents."""
    # Setup mock
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_model_class.return_value = mock_model
    
    # Test
    embedder = Embedder()
    texts = ["test1", "test2"]
    result = embedder.embed_documents(texts)
    
    # Assertions
    assert len(result) == 2
    assert len(result[0]) == 3  # Should match the mock embedding dimension
    mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)

@patch('sentence_transformers.SentenceTransformer')
def test_embed_query(mock_model_class):
    """Test embedding a single query."""
    # Setup mock
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_model_class.return_value = mock_model
    
    # Test
    embedder = Embedder()
    text = "test query"
    result = embedder.embed_query(text)
    
    # Assertions
    assert len(result) == 3  # Should match the mock embedding dimension
    mock_model.encode.assert_called_once_with([text], convert_to_numpy=True)
