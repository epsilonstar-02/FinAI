import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import os

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_embedder():
    """Mock the embedder for testing."""
    with patch('agents.retriever_agent.embedder.embedder') as mock:
        # Mock the embed_documents method to return dummy embeddings
        mock.embed_documents.side_effect = lambda texts: [[0.1] * 384] * len(texts)
        # Mock the embed_query method to return a dummy embedding
        mock.embed_query.return_value = [0.1] * 384
        yield mock

@pytest.fixture
def sample_documents():
    """Generate sample documents for testing."""
    return [
        {
            "page_content": "This is a test document about machine learning.",
            "metadata": {"source": "test_source", "page": 1}
        },
        {
            "page_content": "Another document about artificial intelligence.",
            "metadata": {"source": "test_source", "page": 2}
        },
        {
            "page_content": "A third document about deep learning.",
            "metadata": {"source": "test_source", "page": 3}
        }
    ]

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from agents.retriever_agent.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)
