import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agents.retriever_agent.embedder import Embedder

def test_embedder_singleton():
    """Test that only one instance of Embedder exists."""
    embedder1 = Embedder()
    embedder2 = Embedder()
    assert embedder1 is embedder2

@patch('agents.retriever_agent.embedder.SentenceTransformer')
def test_embed_documents(mock_model_class):
    """Test embedding multiple documents."""
    # Setup mock - using 384 dimensions to match the expected embedding size
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
    mock_model_class.return_value = mock_model
    
    # Important: Reset the singleton instance to force recreation with our mock
    Embedder._instance = None
    Embedder._model = None
    
    # Test
    embedder = Embedder()
    texts = ["test1", "test2"]
    result = embedder.embed_documents(texts)
    
    # Assertions
    assert len(result) == 2
    assert len(result[0]) == 384  # Should match the expected embedding dimension
    mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)

@patch('agents.retriever_agent.embedder.SentenceTransformer')
def test_embed_query(mock_model_class):
    """Test embedding a single query."""
    # Setup mock - using 384 dimensions to match the expected embedding size
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 384])
    mock_model_class.return_value = mock_model
    
    # Important: Reset the singleton instance to force recreation with our mock
    Embedder._instance = None
    Embedder._model = None
    
    # Test
    embedder = Embedder()
    text = "test query"
    result = embedder.embed_query(text)
    
    # Assertions
    assert len(result) == 384  # Should match the expected embedding dimension
    mock_model.encode.assert_called_once_with([text], convert_to_numpy=True)
