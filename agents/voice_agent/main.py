"""FastAPI application for the Voice Agent."""
import io
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from agents.voice_agent.config import settings
from agents.voice_agent.models import STTRequest, STTResponse, TTSRequest, TTSResponse
from agents.voice_agent.stt_client import STTClient, STTClientError
from agents.voice_agent.tts_client import TTSClient, TTSClientError

# Initialize the FastAPI application
app = FastAPI(
    title="Voice Agent",
    description="Voice processing agent for FinAI",
    version="0.1.0",
)

# Initialize clients
stt_client = STTClient()
tts_client = TTSClient()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "agent": "Voice Agent"}


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text."""
    try:
        # Read the audio file
        contents = await file.read()
        
        # Transcribe the audio
        text, confidence = await stt_client.transcribe(contents)
        
        return STTResponse(
            text=text,
            confidence=confidence,
        )
    except STTClientError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech."""
    try:
        # Synthesize speech
        audio_bytes = await tts_client.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except TTSClientError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
