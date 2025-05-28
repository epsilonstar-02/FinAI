"""FastAPI application for the Voice Agent with multi-provider support."""
import io
import time
import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Body, Depends, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from .config import settings, STTProvider, TTSProvider
from .models import (
    STTRequest, 
    STTResponse, 
    TTSRequest, 
    TTSResponse, 
    HealthResponse,
    AvailableProvidersResponse,
    VoiceListResponse
)
from .multi_stt_client import get_multi_stt_client, transcribe, STTError, AllProvidersFailedError, ProviderError as STTProviderError
from .multi_tts_client import get_multi_tts_client, synthesize, TTSError, AllProvidersFailedError as TTSAllProvidersFailedError, ProviderError as TTSProviderError

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI application
app = FastAPI(
    title="Voice Agent",
    description="Multi-provider voice processing agent for FinAI",
    version="0.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
stt_client = get_multi_stt_client()
tts_client = get_multi_tts_client()

# Rate limiting settings
REQUEST_COUNTS = {}
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS = 100        # requests per duration

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean up old entries
    for ip in list(REQUEST_COUNTS.keys()):
        if current_time - REQUEST_COUNTS[ip]["timestamp"] > RATE_LIMIT_DURATION:
            del REQUEST_COUNTS[ip]
    
    # Check if client has exceeded rate limit
    if client_ip in REQUEST_COUNTS:
        request_info = REQUEST_COUNTS[client_ip]
        if current_time - request_info["timestamp"] < RATE_LIMIT_DURATION:
            if request_info["count"] >= MAX_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please try again later."}
                )
            request_info["count"] += 1
        else:
            # Reset if outside the window
            REQUEST_COUNTS[client_ip] = {"count": 1, "timestamp": current_time}
    else:
        # First request from this IP
        REQUEST_COUNTS[client_ip] = {"count": 1, "timestamp": current_time}
    
    # Process the request
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Get available providers
    available_stt_providers = settings.get_available_stt_providers()
    available_tts_providers = settings.get_available_tts_providers()
    
    return HealthResponse(
        status="ok",
        agent="Voice Agent",
        version="0.2.0",
        timestamp=datetime.utcnow(),
        stt_providers=[p.value for p in available_stt_providers],
        tts_providers=[p.value for p in available_tts_providers],
        default_stt_provider=settings.DEFAULT_STT_PROVIDER.value,
        default_tts_provider=settings.DEFAULT_TTS_PROVIDER.value
    )


@app.get("/providers", response_model=AvailableProvidersResponse)
async def get_available_providers():
    """Get available STT and TTS providers."""
    available_stt = settings.get_available_stt_providers()
    available_tts = settings.get_available_tts_providers()
    
    return AvailableProvidersResponse(
        stt={
            "available": [p.value for p in available_stt],
            "default": settings.DEFAULT_STT_PROVIDER.value,
            "fallbacks": [p.value for p in settings.FALLBACK_STT_PROVIDERS if p in available_stt]
        },
        tts={
            "available": [p.value for p in available_tts],
            "default": settings.DEFAULT_TTS_PROVIDER.value,
            "fallbacks": [p.value for p in settings.FALLBACK_TTS_PROVIDERS if p in available_tts]
        }
    )


@app.get("/voices", response_model=VoiceListResponse)
async def get_available_voices(provider: Optional[str] = Query(None, description="TTS provider to get voices for")):
    """Get available voices for TTS providers."""
    # This is a simplified implementation - in a real system, we would
    # query each provider for available voices
    
    # Default voices for each provider
    voices = {
        TTSProvider.PYTTSX3.value: ["default", "male", "female"],
        TTSProvider.GTTS.value: ["en", "en-us", "en-gb", "fr", "de", "es"],
        TTSProvider.EDGE.value: [
            "en-US-AriaNeural", 
            "en-US-GuyNeural", 
            "en-GB-SoniaNeural",
            "en-GB-RyanNeural"
        ],
        TTSProvider.ELEVENLABS.value: ["Rachel", "Domi", "Bella", "Antoni", "Thomas"],
        TTSProvider.AMAZON_POLLY.value: [
            "Joanna", "Matthew", "Salli", "Kimberly", 
            "Kendra", "Joey", "Brian", "Amy"
        ],
        TTSProvider.SILERO.value: ["en_0", "en_1", "en_2"],
        TTSProvider.COQUI.value: ["default"]
    }
    
    if provider:
        try:
            provider_enum = TTSProvider(provider)
            if provider_enum in settings.get_available_tts_providers():
                return VoiceListResponse(
                    provider=provider,
                    voices=voices.get(provider, [])
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider '{provider}' is not available"
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider '{provider}'. Valid providers: {[p.value for p in TTSProvider]}"
            )
    else:
        # Return voices for all available providers
        available_providers = settings.get_available_tts_providers()
        return VoiceListResponse(
            provider="all",
            voices_by_provider={
                p.value: voices.get(p.value, []) 
                for p in available_providers
            }
        )


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    provider: Optional[str] = Query(None, description="Optional specific STT provider to use")
):
    """Convert speech to text with multi-provider support."""
    start_time = time.time()
    
    try:
        # Validate provider if specified
        provider_enum = None
        if provider:
            try:
                provider_enum = STTProvider(provider)
                if provider_enum not in settings.get_available_stt_providers():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Provider '{provider}' is not available. Available providers: {[p.value for p in settings.get_available_stt_providers()]}"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider '{provider}'. Valid providers: {[p.value for p in STTProvider]}"
                )
        
        # Read the audio file
        contents = await file.read()
        
        # Transcribe the audio
        text, confidence = await transcribe(contents, provider_enum)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        return STTResponse(
            text=text,
            confidence=confidence,
            provider=provider if provider else settings.DEFAULT_STT_PROVIDER.value,
            elapsed_time=elapsed_time
        )
    
    except AllProvidersFailedError as e:
        # Return detailed error when all providers fail
        raise HTTPException(
            status_code=502,
            detail={
                "message": "All STT providers failed",
                "provider_errors": e.provider_errors
            }
        )
    
    except STTProviderError as e:
        # Return error for specific provider
        raise HTTPException(
            status_code=502,
            detail=f"Provider error ({e.provider}): {e.message}"
        )
    
    except STTError as e:
        # General STT error
        raise HTTPException(status_code=502, detail=str(e))
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    provider: Optional[str] = Query(None, description="Optional specific TTS provider to use")
):
    """Convert text to speech with multi-provider support."""
    start_time = time.time()
    
    try:
        # Validate provider if specified
        provider_enum = None
        if provider:
            try:
                provider_enum = TTSProvider(provider)
                if provider_enum not in settings.get_available_tts_providers():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Provider '{provider}' is not available. Available providers: {[p.value for p in settings.get_available_tts_providers()]}"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider '{provider}'. Valid providers: {[p.value for p in TTSProvider]}"
                )
        
        # Synthesize speech
        audio_bytes = await synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            provider=provider_enum
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "X-Provider": provider if provider else settings.DEFAULT_TTS_PROVIDER.value,
                "X-Elapsed-Time": str(elapsed_time)
            },
        )
    
    except TTSAllProvidersFailedError as e:
        # Return detailed error when all providers fail
        raise HTTPException(
            status_code=502,
            detail={
                "message": "All TTS providers failed",
                "provider_errors": e.provider_errors
            }
        )
    
    except TTSProviderError as e:
        # Return error for specific provider
        raise HTTPException(
            status_code=502,
            detail=f"Provider error ({e.provider}): {e.message}"
        )
    
    except TTSError as e:
        # General TTS error
        raise HTTPException(status_code=502, detail=str(e))
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
