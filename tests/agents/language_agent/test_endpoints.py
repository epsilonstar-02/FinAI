"""
Unit tests for the FastAPI endpoints of the Language Agent.
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

from agents.language_agent.main import app
from agents.language_agent.models import (
    GenerateRequest, 
    GenerateResponse,
    TemplateRequest,
    TemplateResponse,
    HealthResponse
)
from agents.language_agent.config import LLMProvider


class TestLanguageAgentEndpoints(unittest.TestCase):
    """Test cases for the Language Agent API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test client
        self.client = TestClient(app)
        
        # Patch the generate_text function
        self.generate_text_patch = patch('agents.language_agent.main.generate_text')
        self.mock_generate_text = self.generate_text_patch.start()
        self.mock_generate_text.return_value = "Generated text response"
        
        # Patch the get_template_list function
        self.get_template_list_patch = patch('agents.language_agent.main.get_template_list')
        self.mock_get_template_list = self.get_template_list_patch.start()
        self.mock_get_template_list.return_value = ["template1.j2", "template2.j2"]
        
        # Patch the get_template_content function
        self.get_template_content_patch = patch('agents.language_agent.main.get_template_content')
        self.mock_get_template_content = self.get_template_content_patch.start()
        self.mock_get_template_content.return_value = "Template content: {{ query }}"
        
        # Patch the get_available_models function
        self.get_available_models_patch = patch('agents.language_agent.main.get_available_models')
        self.mock_get_available_models = self.get_available_models_patch.start()
        self.mock_get_available_models.return_value = {
            "llama": "Local LLaMA model",
            "huggingface": "Mistral-7B-Instruct"
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.generate_text_patch.stop()
        self.get_template_list_patch.stop()
        self.get_template_content_patch.stop()
        self.get_available_models_patch.stop()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("providers", data)
        self.assertIn("timestamp", data)
    
    def test_get_models(self):
        """Test the models endpoint."""
        response = self.client.get("/models")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 2)
        self.assertIn("llama", data["models"])
        self.assertIn("huggingface", data["models"])
    
    def test_generate(self):
        """Test the generate endpoint."""
        # Create request payload
        payload = {
            "query": "Tell me about finance",
            "context": ["Financial markets are complex"],
            "provider": "llama"
        }
        
        # Make request to generate endpoint
        response = self.client.post("/generate", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "Generated text response")
        self.assertEqual(data["provider"], "llama")
        
        # Verify generate_text was called with correct arguments
        self.mock_generate_text.assert_called_once()
        # Extract the first positional argument (the prompt)
        prompt_arg = self.mock_generate_text.call_args[0][0]
        self.assertIn("Tell me about finance", prompt_arg)
        self.assertIn("Financial markets are complex", prompt_arg)
    
    def test_generate_with_template(self):
        """Test the generate endpoint with a template."""
        # Create request payload
        payload = {
            "query": "Tell me about finance",
            "context": ["Financial markets are complex"],
            "template_name": "market_brief",
            "provider": "llama"
        }
        
        # Make request to generate endpoint
        response = self.client.post("/generate", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "Generated text response")
        self.assertEqual(data["provider"], "llama")
        
        # Verify generate_text was called
        self.mock_generate_text.assert_called_once()
    
    def test_templates_list(self):
        """Test the templates list endpoint."""
        response = self.client.get("/templates")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("templates", data)
        self.assertEqual(len(data["templates"]), 2)
        self.assertIn("template1.j2", data["templates"])
        self.assertIn("template2.j2", data["templates"])
    
    def test_get_template(self):
        """Test the get template endpoint."""
        response = self.client.get("/templates/template1")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "template1")
        self.assertEqual(data["content"], "Template content: {{ query }}")
        
        # Verify get_template_content was called with correct arguments
        self.mock_get_template_content.assert_called_with("template1")
    
    def test_missing_template(self):
        """Test the get template endpoint with a missing template."""
        # Make get_template_content raise a FileNotFoundError
        self.mock_get_template_content.side_effect = FileNotFoundError("Template not found")
        
        response = self.client.get("/templates/missing_template")
        
        # Check response
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"])
    
    def test_error_handling(self):
        """Test error handling in endpoints."""
        # Make generate_text raise an exception
        self.mock_generate_text.side_effect = ValueError("Test error")
        
        # Create request payload
        payload = {
            "query": "Tell me about finance"
        }
        
        # Make request to generate endpoint
        response = self.client.post("/generate", json=payload)
        
        # Check response
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Test error", data["detail"])


# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
