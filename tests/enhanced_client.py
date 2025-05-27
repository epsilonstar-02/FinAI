"""Enhanced TestClient to handle version compatibility issues."""
from fastapi.testclient import TestClient
from fastapi import FastAPI

def get_test_client(app: FastAPI):
    """Get a test client that works with the current library versions."""
    # FastAPI 0.109.2 uses TestClient(app=app) but can handle TestClient(app)
    # Direct approach is more reliable than complex detection
    return TestClient(app)
