"""Pytest fixtures for orchestrator tests."""
import pytest
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  

from orchestrator.main import app

@pytest.fixture
def test_client():
    """Create a TestClient for the orchestrator FastAPI app.
    
    Using FastAPI's TestClient directly to avoid compatibility issues.
    """
    # Use the standard TestClient directly
    client = TestClient(app)
    return client
