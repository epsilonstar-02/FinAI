"""Speech-to-Text client for the Voice Agent."""
import hashlib
import io
import os
from typing import Tuple, Optional

import httpx
import webrtcvad
import whisper as openai_whisper
from pydub import AudioSegment
try:
    from rnnoise import RNNoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False
    import warnings
    warnings.warn("RNNoise not available. Noise reduction will be disabled.")

from agents.voice_agent.config import settings


class STTClientError(Exception):
    """Exception raised by the STT client."""

    pass


class STTClient:
    """Client for Speech-to-Text operations."""

    def __init__(self):
        """Initialize the STT client."""
        self.vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
        self.model = None
        
        # Initialize RNNoise if enabled and available
        self.rnnoise = None
        if settings.RNNOISE_ENABLED:
            if RNNOISE_AVAILABLE:
                try:
                    self.rnnoise = RNNoise()
                except Exception as e:
                    warnings.warn(f"Failed to initialize RNNoise: {str(e)}. Noise reduction will be disabled.")
            else:
                warnings.warn("RNNoise not available. Noise reduction will be disabled.")

        # Lazy-load the model only when needed
        if settings.STT_PROVIDER == "whisper":
            os.makedirs(os.path.dirname(settings.MODEL_PATH_STT), exist_ok=True)

    def _load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None and settings.STT_PROVIDER == "whisper":
            try:
                self.model = openai_whisper.load_model("base", download_root=settings.MODEL_PATH_STT)
            except Exception as e:
                raise STTClientError(f"Failed to load Whisper model: {str(e)}")

    def _denoise_audio(self, audio_bytes: bytes) -> bytes:
        """Apply noise reduction to audio if enabled."""
        if not settings.RNNOISE_ENABLED or not self.rnnoise:
            return audio_bytes
            
        try:
            # Convert audio to 16kHz mono 16-bit PCM for RNNoise
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)
            
            # Process audio in chunks
            chunk_size = 480  # 10ms at 48kHz
            output = bytearray()
            
            for i in range(0, len(audio.raw_data), chunk_size):
                chunk = audio.raw_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))
                processed = self.rnnoise.process_frame(chunk)
                output.extend(processed)
                
            return bytes(output)
        except Exception as e:
            warnings.warn(f"Error during audio denoising: {str(e)}")
            return audio_bytes

    def _chunk_audio_with_vad(self, audio_bytes: bytes) -> list[bytes]:
        """Split audio into voice-active chunks using WebRTC VAD."""
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Convert to mono, 16kHz, 16-bit PCM for WebRTC VAD
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            
            # Frame size for VAD (30ms)
            frame_size = int(16000 * 0.03)
            
            # Process frames
            is_speech = []
            for i in range(0, len(audio.raw_data), frame_size * 2):
                frame = audio.raw_data[i:i + frame_size * 2]
                if len(frame) == frame_size * 2:  # Ensure complete frame
                    is_speech.append(self.vad.is_speech(frame, 16000))
            
            # Group frames into chunks
            chunks = []
            current_chunk = []
            
            for i, speech in enumerate(is_speech):
                if speech:
                    current_chunk.append(i)
                elif current_chunk:
                    # Add padding frames
                    start = max(0, current_chunk[0] - 5)
                    end = min(len(is_speech) - 1, current_chunk[-1] + 5)
                    
                    # Extract audio chunk
                    chunk_start = start * frame_size * 2
                    chunk_end = (end + 1) * frame_size * 2
                    
                    # Create audio segment for this chunk
                    chunk_audio = AudioSegment(
                        data=audio.raw_data[chunk_start:chunk_end],
                        sample_width=2,
                        frame_rate=16000,
                        channels=1
                    )
                    
                    # Export to bytes
                    output = io.BytesIO()
                    chunk_audio.export(output, format="wav")
                    chunks.append(output.getvalue())
                    
                    current_chunk = []
            
            # Add the last chunk if it exists
            if current_chunk:
                start = max(0, current_chunk[0] - 5)
                end = min(len(is_speech) - 1, current_chunk[-1] + 5)
                
                chunk_start = start * frame_size * 2
                chunk_end = (end + 1) * frame_size * 2
                
                chunk_audio = AudioSegment(
                    data=audio.raw_data[chunk_start:chunk_end],
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                
                output = io.BytesIO()
                chunk_audio.export(output, format="wav")
                chunks.append(output.getvalue())
            
            return chunks if chunks else [audio_bytes]  # Return original if no chunks
        except Exception as e:
            # Fall back to original audio if chunking fails
            return [audio_bytes]

    async def transcribe_whisper(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Transcribe audio using Whisper model."""
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Apply denoising
            processed_audio = self._denoise_audio(audio_bytes)
            
            # Chunk audio using VAD
            chunks = self._chunk_audio_with_vad(processed_audio)
            
            if len(chunks) == 1:
                # Single chunk processing
                audio = AudioSegment.from_file(io.BytesIO(chunks[0]))
                with io.BytesIO() as f:
                    audio.export(f, format="wav")
                    f.seek(0)
                    result = self.model.transcribe(f.read())
                    return result["text"].strip(), result.get("confidence", 0.9)
            else:
                # Multiple chunks processing
                texts = []
                avg_confidence = 0.0
                
                for chunk in chunks:
                    audio = AudioSegment.from_file(io.BytesIO(chunk))
                    with io.BytesIO() as f:
                        audio.export(f, format="wav")
                        f.seek(0)
                        result = self.model.transcribe(f.read())
                        if result["text"].strip():
                            texts.append(result["text"].strip())
                            avg_confidence += result.get("confidence", 0.9)
                
                # Calculate average confidence for all chunks
                if texts:
                    avg_confidence = avg_confidence / len(texts)
                    return " ".join(texts), avg_confidence
                else:
                    return "", 0.0
                
        except Exception as e:
            raise STTClientError(f"Failed to transcribe audio: {str(e)}")

    async def google_stt(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Transcribe audio using Google Speech-to-Text API (free tier)."""
        try:
            # Apply denoising
            processed_audio = self._denoise_audio(audio_bytes)
            
            # Chunk audio using VAD
            chunks = self._chunk_audio_with_vad(processed_audio)
            
            # Process chunks with Google STT
            async with httpx.AsyncClient() as client:
                texts = []
                confidences = []
                
                for chunk in chunks:
                    # Convert to mono, 16kHz, 16-bit PCM for Google STT
                    audio = AudioSegment.from_file(io.BytesIO(chunk))
                    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                    
                    with io.BytesIO() as f:
                        audio.export(f, format="wav")
                        f.seek(0)
                        
                        # Use Google Speech-to-Text REST API
                        response = await client.post(
                            "https://speech.googleapis.com/v1/speech:recognize",
                            params={"key": os.environ.get("GOOGLE_API_KEY")},
                            json={
                                "config": {
                                    "encoding": "LINEAR16",
                                    "sampleRateHertz": 16000,
                                    "languageCode": "en-US",
                                },
                                "audio": {
                                    "content": audio.raw_data.hex(),
                                },
                            },
                            timeout=10,
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("results"):
                                alternative = result["results"][0]["alternatives"][0]
                                texts.append(alternative["transcript"])
                                confidences.append(alternative.get("confidence", 0.9))
                
                if texts:
                    avg_confidence = sum(confidences) / len(confidences)
                    return " ".join(texts), avg_confidence
                else:
                    return "", 0.0
                
        except Exception as e:
            raise STTClientError(f"Failed to transcribe audio with Google STT: {str(e)}")

    async def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Transcribe audio using the configured provider."""
        if settings.STT_PROVIDER == "whisper":
            return await self.transcribe_whisper(audio_bytes)
        elif settings.STT_PROVIDER == "google-stt":
            return await self.google_stt(audio_bytes)
        else:
            raise STTClientError(f"Unsupported STT provider: {settings.STT_PROVIDER}")
