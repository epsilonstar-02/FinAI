"""Pytest fixtures for orchestrator tests."""
import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from orchestrator.main import app
from tests.enhanced_client import get_test_client

@pytest.fixture
def test_client():
    """Create a TestClient for the orchestrator FastAPI app.
    
    This ensures the TestClient is properly initialized to avoid the
    "TypeError: Client.__init__() got an unexpected keyword argument 'app'" error.
    """
    # Use the enhanced client that handles version compatibility issues
    client = get_test_client(app)
    return client
