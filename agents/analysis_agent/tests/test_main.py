import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time 
from pydantic import ValidationError

# Adjust import paths as necessary
from agents.analysis_agent.main import app, settings
from agents.analysis_agent.models import AnalyzeResponse, HistoricalDataPoint, ProviderInfo, RiskMetrics
from agents.analysis_agent.providers import DefaultAnalysisProvider, AdvancedAnalysisProvider

# Note: conftest.py's mock_settings fixture disables cache and rate limiting by default.
# Tests needing these features must re-enable them via monkeypatch.

class TestHealthEndpoint:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok", "agent": "Analysis Agent"}

class TestAnalyzeEndpoint:
    def test_analyze_success_default_provider(self, client: TestClient, sample_analyze_request_payload):
        response = client.post("/analyze", json=sample_analyze_request_payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "exposures" in data
        assert "changes" in data
        assert "volatility" in data
        assert "summary" in data
        assert data["provider_info"]["name"] == "default"
        assert data["correlations"] is None # Not requested
        assert data["risk_metrics"] is None # Not requested

    def test_analyze_success_advanced_provider(self, client: TestClient, sample_analyze_request_payload):
        payload = sample_analyze_request_payload.copy()
        payload["provider"] = "advanced"
        response = client.post("/analyze", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["provider_info"]["name"] == "advanced"

    def test_analyze_include_correlations(self, client: TestClient, sample_analyze_request_payload, monkeypatch):
        monkeypatch.setattr(settings, 'ENABLE_CORRELATION_ANALYSIS', True)
        payload = sample_analyze_request_payload.copy()
        payload["include_correlations"] = True
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "correlations" in data
        assert data["correlations"] is not None
        assert "AAPL" in data["correlations"]
        assert "MSFT" in data["correlations"]["AAPL"]

    def test_analyze_include_risk_metrics_default_provider(self, client: TestClient, sample_analyze_request_payload, monkeypatch, long_historical_data_fixture):
        monkeypatch.setattr(settings, 'ENABLE_RISK_METRICS', True)
        payload = sample_analyze_request_payload.copy()
        payload["include_risk_metrics"] = True
        # Use long_historical_data_fixture for risk metrics
        historical_payload = {
            symbol: [hdp.model_dump() for hdp in hdp_list]
            for symbol, hdp_list in long_historical_data_fixture.items()
        }
        payload["historical"] = historical_payload
        payload["prices"] = {k: 100.0 for k in long_historical_data_fixture.keys()}


        response = client.post("/analyze", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "risk_metrics" in data
        assert data["risk_metrics"] is not None
        assert "SYM1" in data["risk_metrics"]
        # Validate against RiskMetrics model structure (default provider includes all required fields)
        risk_metric_obj = RiskMetrics.model_validate(data["risk_metrics"]["SYM1"])
        assert risk_metric_obj.beta is not None 
        assert risk_metric_obj.var_95 is not None

    def test_analyze_include_risk_metrics_advanced_provider_validation_issue(
        self, client: TestClient, sample_analyze_request_payload, monkeypatch, long_historical_data_fixture
    ):
        monkeypatch.setattr(settings, 'ENABLE_RISK_METRICS', True)
        payload = sample_analyze_request_payload.copy()
        payload["provider"] = "advanced"
        payload["include_risk_metrics"] = True
        historical_payload = {
            symbol: [hdp.model_dump() for hdp in hdp_list]
            for symbol, hdp_list in long_historical_data_fixture.items()
        }
        payload["historical"] = historical_payload
        payload["prices"] = {k: 100.0 for k in long_historical_data_fixture.keys()}

        # Expect the ValidationError to be raised by the TestClient
        # because it occurs during response model validation within FastAPI's processing.
        with pytest.raises(ValidationError) as excinfo:
            client.post("/analyze", json=payload)
        
        # Assert details about the ValidationError
        error_summary = str(excinfo.value)
        # print(f"Validation Error details: {error_summary}") # For debugging
        assert "4 validation errors for AnalyzeResponse" in error_summary
        assert "risk_metrics.SYM1.var_95" in error_summary
        assert "Field required" in error_summary
        assert "risk_metrics.SYM1.beta" in error_summary

    def test_analyze_validation_error(self, client: TestClient):
        payload = {"invalid_field": "some_value"} # Missing required fields
        response = client.post("/analyze", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["status"] == "error"
        assert data["message"] == "Validation error"
        assert "details" in data

    @patch('agents.analysis_agent.providers.DefaultAnalysisProvider.compute_exposures', side_effect=Exception("Primary provider failure"))
    @patch('agents.analysis_agent.providers.AdvancedAnalysisProvider.compute_exposures') # Mock fallback
    def test_analyze_provider_fallback(self, mock_advanced_exposures, mock_default_exposures, client: TestClient, sample_analyze_request_payload, monkeypatch):
        # Ensure fallback provider is different and will be called
        monkeypatch.setattr(settings, 'ANALYSIS_PROVIDER', 'default') # Primary is default
        monkeypatch.setattr(settings, 'FALLBACK_PROVIDERS', ['advanced']) # Fallback is advanced

        # Mock the successful computation by the fallback provider
        mock_advanced_exposures.return_value = {"AAPL": 0.5, "MSFT": 0.5}
        
        # We also need to mock other methods for the fallback provider, or ensure they don't fail
        with patch('agents.analysis_agent.providers.AdvancedAnalysisProvider.compute_changes', return_value={"AAPL": 0.01, "MSFT": 0.01}), \
             patch('agents.analysis_agent.providers.AdvancedAnalysisProvider.compute_volatility', return_value={"AAPL": 0.1, "MSFT": 0.1}):

            response = client.post("/analyze", json=sample_analyze_request_payload)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["provider_info"]["name"] == "advanced" # Fallback provider name
            assert data["provider_info"]["fallback_used"] is True
            mock_default_exposures.assert_called_once()
            mock_advanced_exposures.assert_called_once()

    @patch('agents.analysis_agent.providers.DefaultAnalysisProvider.compute_exposures', side_effect=Exception("Primary provider failure"))
    @patch('agents.analysis_agent.providers.AdvancedAnalysisProvider.compute_exposures', side_effect=Exception("Fallback provider failure"))
    def test_analyze_all_providers_fail(self, mock_advanced_exposures, mock_default_exposures, client: TestClient, sample_analyze_request_payload, monkeypatch):
        monkeypatch.setattr(settings, 'ANALYSIS_PROVIDER', 'default')
        monkeypatch.setattr(settings, 'FALLBACK_PROVIDERS', ['advanced'])
        
        response = client.post("/analyze", json=sample_analyze_request_payload)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Analysis failed with all providers" in data["message"]
        mock_default_exposures.assert_called_once()
        mock_advanced_exposures.assert_called_once()

    def test_analyze_caching(self, client: TestClient, sample_analyze_request_payload, monkeypatch, mock_time):
        monkeypatch.setattr(settings, 'CACHE_ENABLED', True)
        monkeypatch.setattr(settings, 'CACHE_TTL', 3600) # 1 hour
        
        # Clear cache for this test if it's module-scoped
        from agents.analysis_agent.main import analysis_cache
        analysis_cache.clear()

        # First call - should compute and cache
        response1 = client.post("/analyze", json=sample_analyze_request_payload)
        assert response1.status_code == status.HTTP_200_OK
        data1 = response1.json()
        assert len(analysis_cache) == 1

        # Second call - should hit cache
        # To ensure it's a cache hit, we can mock the provider to fail if called again
        with patch('agents.analysis_agent.providers.DefaultAnalysisProvider.compute_exposures', side_effect=AssertionError("Should not be called")):
            response2 = client.post("/analyze", json=sample_analyze_request_payload)
            assert response2.status_code == status.HTTP_200_OK
            data2 = response2.json()
            assert data1 == data2 # Ensure response is the same

        # Advance time beyond TTL
        mock_time.advance_time(settings.CACHE_TTL + 60)

        # Third call - should be a cache miss, recompute
        # Provider mock is no longer active here, so it will compute
        response3 = client.post("/analyze", json=sample_analyze_request_payload)
        assert response3.status_code == status.HTTP_200_OK
        assert len(analysis_cache) == 1 # Old entry replaced or cache cleaned and new one added

    def test_analyze_rate_limiting(self, client: TestClient, sample_analyze_request_payload, monkeypatch, mock_time):
        monkeypatch.setattr(settings, 'RATE_LIMIT_ENABLED', True)
        monkeypatch.setattr(settings, 'RATE_LIMIT', 2) # 2 requests per minute

        from agents.analysis_agent.main import rate_limit_state
        # Reset rate limit state for the test
        rate_limit_state["count"] = 0
        rate_limit_state["reset_time"] = mock_time.time() + 60


        # First two requests should succeed
        response1 = client.post("/analyze", json=sample_analyze_request_payload)
        assert response1.status_code == status.HTTP_200_OK
        
        response2 = client.post("/analyze", json=sample_analyze_request_payload)
        assert response2.status_code == status.HTTP_200_OK

        # Third request should be rate limited
        response3 = client.post("/analyze", json=sample_analyze_request_payload)
        assert response3.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Rate limit exceeded" in response3.json()["message"]

        # Advance time to reset the rate limit
        mock_time.advance_time(61) # Advance by more than 60 seconds

        # Fourth request should succeed again
        response4 = client.post("/analyze", json=sample_analyze_request_payload)
        assert response4.status_code == status.HTTP_200_OK
        
    def test_correlation_analysis_disabled(self, client: TestClient, sample_analyze_request_payload, monkeypatch):
        monkeypatch.setattr(settings, 'ENABLE_CORRELATION_ANALYSIS', False)
        payload = sample_analyze_request_payload.copy()
        payload["include_correlations"] = True # Request it
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["correlations"] is None # Should be None as feature flag is False

    def test_risk_metrics_disabled(self, client: TestClient, sample_analyze_request_payload, monkeypatch):
        monkeypatch.setattr(settings, 'ENABLE_RISK_METRICS', False)
        payload = sample_analyze_request_payload.copy()
        payload["include_risk_metrics"] = True # Request it
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["risk_metrics"] is None # Should be None as feature flag is False

    def test_missing_build_summary_graceful_handling(self, client: TestClient, sample_analyze_request_payload, monkeypatch):
        # Ensure FALLBACK_PROVIDERS is set up so the fallback logic is triggered
        # Assuming 'default' is the primary, and 'advanced' is a fallback.
        # If sample_analyze_request_payload doesn't specify a provider, it uses settings.ANALYSIS_PROVIDER
        # Let's assume settings.ANALYSIS_PROVIDER is 'default' and 'advanced' is in FALLBACK_PROVIDERS.
        # monkeypatch.setattr(settings, 'ANALYSIS_PROVIDER', 'default') # If not already default
        # monkeypatch.setattr(settings, 'FALLBACK_PROVIDERS', ['advanced']) # Ensure 'advanced' is a fallback

        with patch('agents.analysis_agent.main.build_summary', side_effect=NameError("Simulated NameError during build_summary")):
            response = client.post("/analyze", json=sample_analyze_request_payload)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            json_response = response.json()
            
            # The message is from the HTTPException explicitly raised after all fallbacks fail
            # This HTTPException is then handled by http_exception_handler
            expected_message = "Analysis failed with all providers. Original error: Simulated NameError during build_summary"
            assert json_response["status"] == "error"
            assert json_response["message"] == expected_message
            # For an HTTPException, create_error_response sets exc.detail as the message, and details might be None
            # if not explicitly passed as a third argument to create_error_response by http_exception_handler.
            # Your http_exception_handler does: create_error_response(..., message=exc.detail, details=None effectively)
            assert json_response.get("details") is None # Or it might be str(exc) if you change the handler