"""Orchestrator Agent main application."""
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Response, status

from orchestrator import client
from orchestrator.config import settings
from orchestrator.models import RunRequest, StepLog, ErrorLog, RunResponse

app = FastAPI(title="Orchestrator", version="0.1.0")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "agent": "Orchestrator"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    """Run the orchestration process based on input."""
    input_text = request.input.strip()
    mode = request.mode.lower()
    params = request.params
    steps = []
    errors = []
    audio_url = None
    
    # Validate mode
    if mode not in ["text", "voice"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode: {mode}. Must be 'text' or 'voice'."
        )
    
    # Process voice input if mode is voice
    if mode == "voice" and "audio_bytes" in params:
        try:
            import base64
            # Decode base64 audio data
            audio_data = base64.b64decode(params["audio_bytes"])
            latency_ms, stt_result = await client.call_stt(audio_data)
            steps.append(StepLog(tool="voice_agent_stt", latency_ms=latency_ms, response=stt_result))
            input_text = stt_result.get("text", input_text)
        except Exception as e:
            errors.append(ErrorLog(tool="voice_agent_stt", message=str(e)))
    
    # Prepare tasks based on params
    tasks = []
    results = []
    
    # Symbol lookup for financial data
    if "symbols" in params:
        tasks.append(asyncio.create_task(client.call_api(params["symbols"])))
    
    # News scraping
    if "topic" in params:
        limit = params.get("limit", 3)
        tasks.append(asyncio.create_task(client.call_scrape(params["topic"], limit)))
    
    # Vector store retrieval
    if "query" in params:
        k = params.get("k", 5)
        tasks.append(asyncio.create_task(client.call_retrieve(params["query"], k)))
    
    # Price analysis
    if "prices" in params:
        historical = params.get("historical", False)
        tasks.append(asyncio.create_task(client.call_analysis(params["prices"], historical)))
    
    # Execute all tasks concurrently and handle errors
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    context = {}
    for i, result in enumerate(results):
        tool_mapping = {
            0: "api_agent",
            1: "scraping_agent", 
            2: "retriever_agent",
            3: "analysis_agent"
        }
        
        tool_name = tool_mapping.get(i, f"agent_{i}")
        
        if isinstance(result, Exception):
            errors.append(ErrorLog(tool=tool_name, message=str(result)))
        else:
            latency_ms, response = result
            steps.append(StepLog(tool=tool_name, latency_ms=latency_ms, response=response))
            context[tool_name] = response
    
    # Generate narrative response using language agent
    output = "Failed to generate a response."
    try:
        latency_ms, language_result = await client.call_language(input_text, context)
        steps.append(StepLog(tool="language_agent", latency_ms=latency_ms, response=language_result))
        output = language_result.get("text", output)
    except Exception as e:
        errors.append(ErrorLog(tool="language_agent", message=str(e)))
    
    # Generate audio if mode is voice
    if mode == "voice":
        try:
            latency_ms, tts_result = await client.call_tts(output)
            steps.append(StepLog(tool="voice_agent_tts", latency_ms=latency_ms, response=tts_result))
            audio_url = tts_result.get("audio_url")
        except Exception as e:
            errors.append(ErrorLog(tool="voice_agent_tts", message=str(e)))
    
    # Return partial results if some operations failed
    if errors and steps:
        return Response(
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            content=RunResponse(
                output=output,
                steps=steps,
                errors=errors,
                audio_url=audio_url
            ).model_dump_json(),
            media_type="application/json"
        )
    
    # Return error if all operations failed
    if errors and not steps:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="All downstream service calls failed"
        )
    
    return RunResponse(
        output=output,
        steps=steps,
        errors=errors,
        audio_url=audio_url
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=8004, reload=True)
