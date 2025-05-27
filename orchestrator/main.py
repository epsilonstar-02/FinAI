"""Orchestrator Agent main application."""
import asyncio
import re
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException

from orchestrator.client import api_client, scraping_client, retriever_client
from orchestrator.models import RunRequest, RunStep, RunResponse

app = FastAPI(title="Orchestrator Agent")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "agent": "Orchestrator Agent"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    """Run the orchestration process based on input."""
    user_input = request.input.strip()
    steps = []

    # Extract potential stock symbols
    symbol_match = re.search(r'\b[A-Z]{1,5}\b', user_input)
    symbol = symbol_match.group(0) if symbol_match else None

    # Extract potential topics from input
    topic = user_input.lower()
    
    # Execute tasks concurrently
    tasks = []
    
    if symbol:
        tasks.append(
            asyncio.create_task(
                _call_api_agent(symbol, "api_agent")
            )
        )
    
    tasks.append(
        asyncio.create_task(
            _call_scraping_agent(topic, 3, "scraping_agent")
        )
    )
    
    tasks.append(
        asyncio.create_task(
            _call_retriever_agent(user_input, 5, "retriever_agent")
        )
    )
    
    # Gather all results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, RunStep):
            steps.append(result)
    
    # Generate output based on gathered information
    output = _generate_response(user_input, steps)
    
    return RunResponse(output=output, steps=steps)


async def _call_api_agent(symbol: str, tool_name: str) -> RunStep:
    """Call the API agent to get price data."""
    try:
        response = await api_client.get(f"/price", params={"symbol": symbol}, tool_name=tool_name)
        return RunStep(tool=tool_name, response=response)
    except HTTPException as e:
        # Re-raise the exception to be caught by the main handler
        raise e


async def _call_scraping_agent(topic: str, limit: int, tool_name: str) -> RunStep:
    """Call the scraping agent to get news data."""
    try:
        response = await scraping_client.get(
            f"/scrape/news", 
            params={"topic": topic, "limit": limit},
            tool_name=tool_name
        )
        return RunStep(tool=tool_name, response=response)
    except HTTPException as e:
        # Re-raise the exception to be caught by the main handler
        raise e


async def _call_retriever_agent(query: str, k: int, tool_name: str) -> RunStep:
    """Call the retriever agent to get relevant information."""
    try:
        response = await retriever_client.get(
            f"/retrieve", 
            params={"q": query, "k": k},
            tool_name=tool_name
        )
        return RunStep(tool=tool_name, response=response)
    except HTTPException as e:
        # Re-raise the exception to be caught by the main handler
        raise e


def _generate_response(user_input: str, steps: List[RunStep]) -> str:
    """Generate a response based on the gathered information."""
    # Basic response generation logic
    response_parts = []
    
    for step in steps:
        if step.tool == "api_agent" and step.response:
            price_data = step.response
            if "price" in price_data:
                response_parts.append(
                    f"Current price for {price_data.get('symbol', 'the stock')}: "
                    f"${price_data.get('price', 'N/A')}"
                )
        
        elif step.tool == "scraping_agent" and step.response:
            news_data = step.response
            if news_data and isinstance(news_data, list) and len(news_data) > 0:
                response_parts.append("Recent news:")
                for idx, article in enumerate(news_data[:3], 1):
                    title = article.get("title", "No title")
                    response_parts.append(f"  {idx}. {title}")
        
        elif step.tool == "retriever_agent" and step.response:
            retriever_data = step.response
            if retriever_data and isinstance(retriever_data, list) and len(retriever_data) > 0:
                response_parts.append("Related information:")
                for idx, item in enumerate(retriever_data[:2], 1):
                    content = item.get("content", "No content")
                    # Truncate long content
                    if len(content) > 100:
                        content = content[:97] + "..."
                    response_parts.append(f"  {idx}. {content}")
    
    if not response_parts:
        return "I couldn't find relevant information for your query."
    
    return "\n\n".join(response_parts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)
