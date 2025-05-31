# agents/voice_agent/main.py

"""FastAPI application for the Voice Agent with multi-provider support."""
import io
import time
import logging
import asyncio # Not directly used in endpoints, but good for type hints
from typing import Optional, List, Dict, Any
from datetime import datetime
import base64 # For base64 response format

from fastapi import (
    FastAPI, File, HTTPException, UploadFile, Query, Body, Depends, Request, status, Path as FastApiPath
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder # For detailed error responses
from pydantic import ValidationError # To catch Pydantic errors

from .config import settings, STTProvider, TTSProvider # Enums for query param validation
from .models import (
    # STTRequest model is not directly used for /stt if File is UploadFile
    STTResponse, 
    TTSRequest, TTSResponse, 
    HealthResponse,
    AvailableProvidersResponse,
    VoiceListResponse
)
# Using aliased public functions from multi-clients
from .multi_stt_client import (
    get_multi_stt_client, transcribe as transcribe_audio, 
    STTError, AllProvidersFailedError as STTAllProvidersFailedError, ProviderError as STTProviderError
)
from .multi_tts_client import (
    get_multi_tts_client, synthesize as synthesize_speech, 
    TTSError, AllProvidersFailedError as TTSAllProvidersFailedError, ProviderError as TTSProviderError
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Agent",
    description="Multi-provider voice processing (STT/TTS) agent for FinAI.",
    version="0.3.0", # Updated version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Client singletons (initialized at startup)
_stt_client_instance: Optional['MultiSTTClient'] = None
_tts_client_instance: Optional['MultiTTSClient'] = None

@app.on_event("startup")
async def startup_event():
    global _stt_client_instance, _tts_client_instance
    try:
        _stt_client_instance = get_multi_stt_client()
        logger.info(f"MultiSTTClient initialized. Available STT: {[p.value for p in _stt_client_instance.available_providers]}")
    except Exception as e:
        logger.critical(f"MultiSTTClient failed to initialize: {e}", exc_info=True)
    
    try:
        _tts_client_instance = get_multi_tts_client()
        logger.info(f"MultiTTSClient initialized. Available TTS: {[p.value for p in _tts_client_instance.available_providers]}")
    except Exception as e:
        logger.critical(f"MultiTTSClient failed to initialize: {e}", exc_info=True)
    
    logger.info("Voice Agent startup complete.")

# Dependencies to get clients
def get_stt_client_dependency() -> 'MultiSTTClient':
    if _stt_client_instance is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="STT Client not initialized.")
    return _stt_client_instance

def get_tts_client_dependency() -> 'MultiTTSClient':
    if _tts_client_instance is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTS Client not initialized.")
    return _tts_client_instance


# Rate limiting (same as Language Agent's refined version)
_REQUEST_COUNTS_VOICE: Dict[str, List[float]] = {}
_RATE_LIMIT_WINDOW_SECONDS_VOICE = 60
_MAX_REQUESTS_PER_WINDOW_VOICE = 100 # Example limit

@app.middleware("http")
async def rate_limit_middleware_voice(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown_ip_voice"
    current_time = time.time()
    
    ip_timestamps = _REQUEST_COUNTS_VOICE.get(client_ip, [])
    ip_timestamps = [ts for ts in ip_timestamps if current_time - ts < _RATE_LIMIT_WINDOW_SECONDS_VOICE]
    
    if len(ip_timestamps) >= _MAX_REQUESTS_PER_WINDOW_VOICE:
        logger.warning(f"Rate limit exceeded for IP (Voice Agent): {client_ip}")
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            content={"detail": "Too many requests. Please try again later."})
    
    ip_timestamps.append(current_time)
    _REQUEST_COUNTS_VOICE[client_ip] = ip_timestamps
    
    response = await call_next(request)
    return response

# --- Exception Handlers ---
async def base_provider_error_handler(request: Request, exc_type: Any, status_code: int, base_message: str):
    error_content = {"detail": base_message, "message": str(exc_type)}
    if hasattr(exc_type, 'provider_errors'): # For AllProvidersFailedError
        error_content["provider_errors"] = exc_type.provider_errors
    elif hasattr(exc_type, 'provider') and hasattr(exc_type, 'message'): # For ProviderError
        error_content["provider_specific_error"] = {exc_type.provider: exc_type.message}
    
    logger.error(f"{base_message} for {request.url.path}: {exc_type}", exc_info=False) # exc_info might be too verbose for provider errors often
    return JSONResponse(status_code=status_code, content=jsonable_encoder(error_content))

@app.exception_handler(STTAllProvidersFailedError)
async def stt_all_providers_failed_handler(r: Request, exc: STTAllProvidersFailedError):
    return await base_provider_error_handler(r, exc, status.HTTP_502_BAD_GATEWAY, "All STT providers failed.")
@app.exception_handler(STTProviderError)
async def stt_provider_error_handler(r: Request, exc: STTProviderError):
    return await base_provider_error_handler(r, exc, status.HTTP_502_BAD_GATEWAY, f"STT Provider error ({exc.provider}).")
@app.exception_handler(STTError)
async def stt_general_error_handler(r: Request, exc: STTError):
    return await base_provider_error_handler(r, exc, status.HTTP_500_INTERNAL_SERVER_ERROR, "A general STT error occurred.")

@app.exception_handler(TTSAllProvidersFailedError)
async def tts_all_providers_failed_handler(r: Request, exc: TTSAllProvidersFailedError):
    return await base_provider_error_handler(r, exc, status.HTTP_502_BAD_GATEWAY, "All TTS providers failed.")
@app.exception_handler(TTSProviderError)
async def tts_provider_error_handler(r: Request, exc: TTSProviderError):
    return await base_provider_error_handler(r, exc, status.HTTP_502_BAD_GATEWAY, f"TTS Provider error ({exc.provider}).")
@app.exception_handler(TTSError)
async def tts_general_error_handler(r: Request, exc: TTSError):
    return await base_provider_error_handler(r, exc, status.HTTP_500_INTERNAL_SERVER_ERROR, "A general TTS error occurred.")

@app.exception_handler(ValidationError) # Pydantic validation errors
async def pydantic_validation_error_handler(r: Request, exc: ValidationError):
    logger.warning(f"Request validation error for {r.url.path}: {exc.errors()}")
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=jsonable_encoder({"detail": "Request validation failed.", "errors": exc.errors()}))

@app.exception_handler(Exception) # Generic fallback
async def generic_exception_handler_voice(r: Request, exc: Exception):
    logger.error(f"Unhandled exception for {r.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder({"detail": "An unexpected internal server error occurred.", "message": str(exc)}))
# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check_endpoint(
    stt_client: 'MultiSTTClient' = Depends(get_stt_client_dependency),
    tts_client: 'MultiTTSClient' = Depends(get_tts_client_dependency)
):
    return HealthResponse(
        status="ok" if stt_client and tts_client else "degraded",
        agent="Voice Agent",
        version=app.version,
        timestamp=datetime.utcnow(),
        stt_providers=[p.value for p in stt_client.available_providers] if stt_client else [],
        tts_providers=[p.value for p in tts_client.available_providers] if tts_client else [],
        default_stt_provider=settings.DEFAULT_STT_PROVIDER.value,
        default_tts_provider=settings.DEFAULT_TTS_PROVIDER.value
    )

@app.get("/providers", response_model=AvailableProvidersResponse, tags=["Utility"])
async def get_available_providers_endpoint(
    stt_client: 'MultiSTTClient' = Depends(get_stt_client_dependency),
    tts_client: 'MultiTTSClient' = Depends(get_tts_client_dependency)
):
    available_stt = stt_client.available_providers if stt_client else []
    available_tts = tts_client.available_providers if tts_client else []
    
    return AvailableProvidersResponse(
        stt=ProviderInfo(
            available=[p.value for p in available_stt],
            default=settings.DEFAULT_STT_PROVIDER.value,
            fallbacks=[p.value for p in settings.FALLBACK_STT_PROVIDERS if p in available_stt]
        ),
        tts=ProviderInfo(
            available=[p.value for p in available_tts],
            default=settings.DEFAULT_TTS_PROVIDER.value,
            fallbacks=[p.value for p in settings.FALLBACK_TTS_PROVIDERS if p in available_tts]
        )
    )

@app.get("/voices", response_model=VoiceListResponse, tags=["TTS Utility"])
async def get_available_voices_endpoint(
    tts_client: 'MultiTTSClient' = Depends(get_tts_client_dependency), # Added client dep
    provider_query: Optional[str] = Query(None, alias="provider", description=f"Optional: TTS provider name (e.g., {', '.join([p.value for p in TTSProvider])})")
):
    # This remains a simplified placeholder. Real implementation would query each provider.
    # Example: if provider_query == TTSProvider.ELEVENLABS.value and ELEVENLABS_AVAILABLE:
    # voices_list = await asyncio.get_event_loop().run_in_executor(None, elevenlabs_voices)
    # return VoiceListResponse(provider=provider_query, voices=[v.name for v in voices_list if v.voice_id and v.name])
    
    all_voices_map = { # Hardcoded example, extend this
        TTSProvider.PYTTSX3.value: ["Default Male", "Default Female"], # pyttsx3 voices are system-dependent
        TTSProvider.GTTS.value: ["en-US", "en-GB", "es-ES", "fr-FR"], # Lang codes for gTTS
        TTSProvider.EDGE.value: ["en-US-AriaNeural", "en-US-GuyNeural", "de-DE-KatjaNeural"],
        TTSProvider.ELEVENLABS.value: [settings.ELEVENLABS_VOICE_ID or "Rachel", "Bella", "Adam"],
        TTSProvider.AMAZON_POLLY.value: [settings.POLLY_VOICE_ID or "Joanna", "Matthew", "Salli"],
        TTSProvider.SILERO.value: [settings.SILERO_SPEAKER or "en_0", "en_117"], # Example Silero speaker IDs
        TTSProvider.COQUI.value: ["default_coqui_voice"], # Coqui often uses model-specific default or speaker embeddings
    }
    
    available_tts_providers = tts_client.available_providers if tts_client else []

    if provider_query:
        try:
            provider_enum = TTSProvider(provider_query.lower())
            if provider_enum not in available_tts_providers:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Provider '{provider_query}' not available.")
            return VoiceListResponse(provider=provider_enum.value, voices=all_voices_map.get(provider_enum.value, ["default"]))
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid provider name '{provider_query}'.")
    else:
        return VoiceListResponse(
            provider="all",
            voices_by_provider={
                p.value: all_voices_map.get(p.value, ["default"]) for p in available_tts_providers
            }
        )


@app.post("/stt", response_model=STTResponse, tags=["STT"])
async def speech_to_text_endpoint(
    stt_client: 'MultiSTTClient' = Depends(get_stt_client_dependency),
    file: UploadFile = File(..., description="Audio file to transcribe (e.g., WAV, MP3)"),
    provider_query: Optional[str] = Query(None, alias="provider", description="Optional: STT provider name"),
    language: Optional[str] = Query(None, description="Optional: Language code (e.g., 'en-US')")
):
    start_time_ns = time.perf_counter_ns()
    
    provider_enum_preference: Optional[STTProvider] = None
    if provider_query:
        try:
            provider_enum_preference = STTProvider(provider_query.lower())
            if provider_enum_preference not in stt_client.available_providers:
                valid_opts = ", ".join([p.value for p in stt_client.available_providers])
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Requested STT provider '{provider_query}' not available. Available: [{valid_opts}]")
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid STT provider name '{provider_query}'.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    # `transcribe_audio` is the imported public function from multi_stt_client
    text, confidence = await transcribe_audio(
        audio_bytes, provider_preference=provider_enum_preference, language=language
    )
    
    elapsed_time_s = (time.perf_counter_ns() - start_time_ns) / 1_000_000_000
    
    return STTResponse(
        text=text,
        confidence=confidence,
        provider=(provider_enum_preference.value if provider_enum_preference else settings.DEFAULT_STT_PROVIDER.value), # Approximates actual used provider
        elapsed_time=round(elapsed_time_s, 4),
        language=language # Or detected language if STT client returns it
    )

@app.post("/tts", tags=["TTS"]) # Response handled by StreamingResponse or JSONResponse
async def text_to_speech_endpoint(
    tts_client: 'MultiTTSClient' = Depends(get_tts_client_dependency),
    request_body: TTSRequest = Body(...), # Use Pydantic model for request body
    provider_query: Optional[str] = Query(None, alias="provider", description="Optional: TTS provider name"),
    response_format: str = Query("stream", description="Response format: 'stream' (audio file) or 'base64' (JSON with base64 audio)", pattern="^(stream|base64)$")
):
    start_time_ns = time.perf_counter_ns()
    
    provider_enum_preference: Optional[TTSProvider] = None
    if provider_query:
        try:
            provider_enum_preference = TTSProvider(provider_query.lower())
            if provider_enum_preference not in tts_client.available_providers:
                valid_opts = ", ".join([p.value for p in tts_client.available_providers])
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Requested TTS provider '{provider_query}' not available. Available: [{valid_opts}]")
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid TTS provider name '{provider_query}'.")

    # `synthesize_speech` is the imported public function
    audio_bytes = await synthesize_speech(
        text=request_body.text,
        voice=request_body.voice,
        speed=request_body.speed,
        provider_preference=provider_enum_preference,
        language=request_body.language
    )
    
    elapsed_time_s = (time.perf_counter_ns() - start_time_ns) / 1_000_000_000
    used_provider_str = provider_enum_preference.value if provider_enum_preference else settings.DEFAULT_TTS_PROVIDER.value # Approximation

    if response_format.lower() == "base64":
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return TTSResponse(
            audio_base64=audio_b64,
            provider=used_provider_str,
            format="mp3", # Assuming MP3 output from clients
            elapsed_time=round(elapsed_time_s, 4)
        )
    else: # stream
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg", # Assuming MP3
            headers={
                "Content-Disposition": "attachment; filename=synthesized_speech.mp3",
                "X-TTS-Provider": used_provider_str,
                "X-Elapsed-Time-Seconds": str(round(elapsed_time_s, 4))
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agents.voice_agent.main:app", host="0.0.0.0", port=8006, reload=True, log_level=settings.LOG_LEVEL.lower())