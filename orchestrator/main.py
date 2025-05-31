# orchestrator/main.py
"""Orchestrator Agent main application."""
import asyncio
import base64 # Moved to top
import re # Keep if specific regex operations are planned, currently unused
from typing import Dict, List, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Response, status

# Use the refactored client and its global instance
from orchestrator.client import agent_client # Renamed from 'client' to 'agent_client' for clarity
from orchestrator.config import settings
from orchestrator.models import RunRequest, StepLog, ErrorLog, RunResponse

app = FastAPI(title="Orchestrator", version="0.1.0")


@app.get("/health", tags=["Utility"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "agent": "Orchestrator"}


@app.post("/run", response_model=RunResponse, tags=["Orchestration"])
async def run(request: RunRequest) -> RunResponse: # FastAPI automatically returns Response or RunResponse
    """Run the orchestration process based on input."""
    input_text = request.input.strip()
    mode = request.mode.lower()
    # Ensure params is always a dict, even if not provided in request
    params = request.params if request.params is not None else {} 
    
    steps: List[StepLog] = []
    errors: List[ErrorLog] = []
    audio_url: Optional[str] = None
    audio_output_b64: Optional[str] = None
    
    # Validate mode
    if mode not in ["text", "voice"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode: {mode}. Must be 'text' or 'voice'."
        )
    
    # Extract model preferences from params
    llm_model = params.get("model")
    stt_provider = params.get("stt_provider")
    tts_provider = params.get("tts_provider")
    voice = params.get("voice")
    speaking_rate = params.get("speaking_rate")
    pitch = params.get("pitch")
    temperature = params.get("temperature")
    max_tokens = params.get("max_tokens")
    
    # Process voice input if mode is voice
    if mode == "voice" and "audio_bytes_b64" in params: # Changed from "audio_bytes" to "audio_bytes_b64" for clarity
        try:
            audio_data = base64.b64decode(params["audio_bytes_b64"])
            latency_ms, stt_result = await agent_client.call_stt(audio_data, provider=stt_provider)
            steps.append(StepLog(tool="voice_agent_stt", latency_ms=latency_ms, response=stt_result))

            if not stt_result.get("success", True) or not stt_result.get("text", "").strip():
                error_message = stt_result.get("detail", "STT failed or returned empty text.")
                logger.warning(f"STT Error/Empty: {error_message}")
                errors.append(ErrorLog(tool="voice_agent_stt", message=error_message))
                
                clarification_text = "I couldn't understand the audio. Could you please try again or provide your question as text?"
                if mode == "voice": # This condition is redundant here, already in voice mode block
                    try:
                        tts_latency_ms, tts_clarify_result = await agent_client.call_tts(
                            clarification_text, provider=tts_provider, voice=voice,
                            speaking_rate=speaking_rate, pitch=pitch
                        )
                        steps.append(StepLog(tool="voice_agent_tts_clarify", latency_ms=tts_latency_ms, response=tts_clarify_result))
                        if tts_clarify_result.get("success", True):
                            audio_output_b64 = tts_clarify_result.get("audio_base64")
                            # audio_url = tts_clarify_result.get("audio_url") # If TTS provides URL
                        else:
                            errors.append(ErrorLog(tool="voice_agent_tts_clarify", message=tts_clarify_result.get("detail", "TTS for clarification failed.")))
                    except Exception as e_tts_clarify:
                        logger.error(f"TTS for clarification failed: {e_tts_clarify}", exc_info=True)
                        errors.append(ErrorLog(tool="voice_agent_tts_clarify", message=str(e_tts_clarify)))
                
                # Return early with clarification
                return RunResponse(
                    output=clarification_text, steps=steps, errors=errors,
                    audio_output_b64=audio_output_b64 #, audio_url=audio_url
                )
            input_text = stt_result.get("text", input_text) # Update input_text with transcription
                
        except base64.binascii.Error as b64_error:
            logger.error(f"Base64 decoding error for audio_bytes_b64: {b64_error}", exc_info=True)
            errors.append(ErrorLog(tool="voice_input_processing", message=f"Invalid base64 audio data: {b64_error}"))
            # Fall through to use original input_text, or raise HTTP 400 if audio was critical
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 audio data provided.")
        except Exception as e_stt: # Catch other STT call related errors
            logger.error(f"Error in STT processing: {e_stt}", exc_info=True)
            errors.append(ErrorLog(tool="voice_agent_stt", message=str(e_stt)))
            # Decide if to continue or return early. For now, let's assume if STT fails hard, use original text.
            # If input_text was *only* from audio, this might be problematic.

    # Prepare agent calls based on input and params
    api_task, scraping_task, retriever_task, analysis_task = None, None, None, None
    context_data: Dict[str, Any] = {} # Renamed from context to avoid conflict with FastAPI's context vars
    
    # Symbol lookup for financial data
    if "symbols" in params:
        api_task = agent_client.call_api(params["symbols"])
    
    # News scraping
    scraping_topic = params.get("topic")
    if scraping_topic:
        limit = params.get("limit", 3) # Default limit for scraping
        scraping_task = agent_client.call_scrape(scraping_topic, limit)
    
    # Vector store retrieval
    retriever_query = params.get("query", input_text) # Default to main input_text
    k_retrieval = params.get("k", 5) # Default k for retrieval
    if retriever_query: # Ensure retriever_query is not empty
        retriever_task = agent_client.call_retrieve(retriever_query, k_retrieval)
    
    # Price analysis
    if "prices" in params:
        analysis_params = {
            "prices": params["prices"],
            "historical": params.get("historical", {}), # Default to empty dict
            "provider": params.get("analysis_provider"),
            "include_correlations": params.get("include_correlations", False),
            "include_risk_metrics": params.get("include_risk_metrics", False)
        }
        analysis_task = agent_client.call_analysis(**analysis_params)
    
    # Execute tasks concurrently
    tasks_to_run_with_names = []
    if api_task: tasks_to_run_with_names.append(("api_agent", api_task))
    if scraping_task: tasks_to_run_with_names.append(("scraping_agent", scraping_task))
    if retriever_task: tasks_to_run_with_names.append(("retriever_agent", retriever_task))
    if analysis_task: tasks_to_run_with_names.append(("analysis_agent", analysis_task))
    
    if tasks_to_run_with_names:
        task_coroutines = [t[1] for t in tasks_to_run_with_names]
        task_names = [t[0] for t in tasks_to_run_with_names]
        
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        for i, result_or_exc in enumerate(results):
            tool_name = task_names[i]
            if isinstance(result_or_exc, Exception):
                logger.error(f"Error from {tool_name}: {result_or_exc}", exc_info=True)
                errors.append(ErrorLog(tool=tool_name, message=str(result_or_exc)))
            else: # Result is (latency_ms, response_dict)
                latency_ms, response_data = result_or_exc
                steps.append(StepLog(tool=tool_name, latency_ms=latency_ms, response=response_data))
                if response_data.get("success", True): # Check for success flag from agent
                    context_data[tool_name] = response_data
                else:
                    errors.append(ErrorLog(tool=tool_name, message=response_data.get("detail", "Agent call reported failure.")))
    
    # Prepare LLM parameters
    llm_call_params: Dict[str, Any] = {}
    if llm_model: llm_call_params["model"] = llm_model
    if temperature is not None: llm_call_params["temperature"] = temperature
    if max_tokens is not None: llm_call_params["max_tokens"] = max_tokens
    
    # Generate narrative response using language agent
    final_output = "Failed to generate a response."
    try:
        lang_latency_ms, language_result = await agent_client.call_language(
            input_text, context_data, **llm_call_params
        )
        steps.append(StepLog(tool="language_agent", latency_ms=lang_latency_ms, response=language_result))
        
        if language_result.get("success", True):
            final_output = language_result.get("text", final_output)
            confidence = language_result.get("confidence", 1.0) # Default to high confidence
            
            # Low confidence fallback retrieval
            if confidence < settings.RETRIEVAL_CONFIDENCE_THRESHOLD and \
               "retriever_agent" not in context_data and \
               not params.get("query"): # Avoid re-retrieving if query was explicit in params
                logger.info(f"Low LLM confidence ({confidence}), attempting fallback retrieval.")
                try:
                    # Use a slightly higher k for fallback
                    fb_retrieve_latency, fb_retrieve_result = await agent_client.call_retrieve(input_text, k=k_retrieval + 3)
                    steps.append(StepLog(tool="retriever_agent_fallback", latency_ms=fb_retrieve_latency, response=fb_retrieve_result))
                    if fb_retrieve_result.get("success", True):
                        context_data["retriever_agent_fallback"] = fb_retrieve_result
                        
                        # Re-call language agent with new context
                        impr_latency, impr_result = await agent_client.call_language(
                            input_text, context_data, **llm_call_params
                        )
                        steps.append(StepLog(tool="language_agent_fallback", latency_ms=impr_latency, response=impr_result))
                        if impr_result.get("success", True) and impr_result.get("text"):
                            final_output = impr_result.get("text")
                        else:
                             errors.append(ErrorLog(tool="language_agent_fallback", message=impr_result.get("detail", "Fallback LLM call failed.")))
                    else:
                        errors.append(ErrorLog(tool="retriever_agent_fallback", message=fb_retrieve_result.get("detail", "Fallback retrieval failed.")))
                except Exception as e_fb_retrieve:
                    logger.error(f"Fallback retrieval/language generation failed: {e_fb_retrieve}", exc_info=True)
                    errors.append(ErrorLog(tool="fallback_processing", message=str(e_fb_retrieve)))
        else:
            errors.append(ErrorLog(tool="language_agent", message=language_result.get("detail", "Language agent call failed.")))

    except Exception as e_lang:
        logger.error(f"Language agent processing failed: {e_lang}", exc_info=True)
        errors.append(ErrorLog(tool="language_agent", message=str(e_lang)))
    
    # Generate audio if mode is voice
    if mode == "voice":
        try:
            tts_latency, tts_result = await agent_client.call_tts(
                final_output, provider=tts_provider, voice=voice,
                speaking_rate=speaking_rate, pitch=pitch
            )
            steps.append(StepLog(tool="voice_agent_tts", latency_ms=tts_latency, response=tts_result))
            if tts_result.get("success", True):
                audio_output_b64 = tts_result.get("audio_base64")
                # audio_url = tts_result.get("audio_url") # If provided
            else:
                errors.append(ErrorLog(tool="voice_agent_tts", message=tts_result.get("detail", "TTS call failed.")))
        except Exception as e_tts:
            logger.error(f"TTS generation failed: {e_tts}", exc_info=True)
            errors.append(ErrorLog(tool="voice_agent_tts", message=str(e_tts)))
    
    response_payload = RunResponse(
        output=final_output, steps=steps, errors=errors,
        audio_output_b64=audio_output_b64 #, audio_url=audio_url
    )

    # Determine final status code
    if errors and not steps: # All critical operations failed
        # This is an internal decision, 502 might be too strong if some initial steps (like STT) worked
        # but then all core data fetching/LLM failed.
        # Let's use 500 if there are errors AND the output is the default "Failed to generate..."
        if final_output == "Failed to generate a response.":
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Core processing failed, unable to generate a meaningful response."
            )
        # Otherwise, even with errors, if there's some output, partial content might be better.
    
    if errors: # Some errors occurred, but some steps might have succeeded
        return Response(
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            content=response_payload.model_dump_json(),
            media_type="application/json"
        )
    
    return response_payload


if __name__ == "__main__":
    import uvicorn
    # Corrected uvicorn run command for module structure
    uvicorn.run("orchestrator.main:app", host=settings.HOST, port=settings.PORT, reload=True)