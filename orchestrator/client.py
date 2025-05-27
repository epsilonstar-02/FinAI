"""HTTP client for calling downstream services."""
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_fixed

from orchestrator.config import settings


class AgentClient:
    """Client for making HTTP requests to agent services with retries and timeouts."""

    def __init__(self, base_url: str, timeout: int = None):
        """Initialize the agent client."""
        self.base_url = base_url
        self.timeout = timeout or settings.TIMEOUT

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        tool_name: str = "unknown",
    ) -> Any:
        """Make an HTTP request to the agent service with retries."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error calling {tool_name}: {str(e)}",
            )

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, tool_name: str = "unknown"
    ) -> Any:
        """Make a GET request to the agent service."""
        return await self._request("GET", endpoint, params=params, tool_name=tool_name)

    async def post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        tool_name: str = "unknown",
    ) -> Any:
        """Make a POST request to the agent service."""
        return await self._request(
            "POST", endpoint, params=params, json_data=json_data, tool_name=tool_name
        )


# Initialize clients for each agent
api_client = AgentClient(settings.API_AGENT_URL)
scraping_client = AgentClient(settings.SCRAPING_AGENT_URL)
retriever_client = AgentClient(settings.RETRIEVER_AGENT_URL)
