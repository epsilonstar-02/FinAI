"""Text-to-Speech client for the Voice Agent."""
import hashlib
import io
import os
from typing import Optional, Union

import diskcache
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment

from agents.voice_agent.config import settings


class TTSClientError(Exception):
    """Exception raised by the TTS client."""

    pass


class TTSClient:
    """Client for Text-to-Speech operations."""

    def __init__(self):
        """Initialize the TTS client."""
        # Initialize cache
        os.makedirs(settings.CACHE_DIR, exist_ok=True)
        self.cache = diskcache.Cache(settings.CACHE_DIR)
        
        # Initialize Gemini client if needed
        if settings.TTS_PROVIDER == "gemini-tts":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise TTSClientError("GOOGLE_API_KEY environment variable is required for Gemini TTS")
            genai.configure(api_key=api_key)

    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate a cache key for TTS request."""
        key_string = f"{text}|{voice}|{speed}"
        return hashlib.md5(key_string.encode()).hexdigest()

    async def synthesize_gtts(self, text: str, voice: str = "default", speed: float = 1.0) -> bytes:
        """Synthesize speech using Google Text-to-Speech."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, voice, speed)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Use gTTS to generate speech
            tts = gTTS(text=text, lang="en", slow=False)
            
            # Save to BytesIO
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Load with pydub
                audio = AudioSegment.from_file(mp3_fp, format="mp3")
                
                # Adjust speed by modifying frame rate
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed)
                })
                
                # Export back to bytes
                output = io.BytesIO()
                audio.export(output, format="mp3")
                result = output.getvalue()
            else:
                result = mp3_fp.getvalue()
            
            # Cache the result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            raise TTSClientError(f"Failed to synthesize speech with gTTS: {str(e)}")

    async def gemini_tts(self, text: str, voice: str = "default", speed: float = 1.0) -> bytes:
        """Synthesize speech using Google Gemini TTS model."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, voice, speed)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Use Gemini model for TTS
            model = genai.GenerativeModel('gemini-tts')
            
            # Generate audio
            response = model.generate_content(
                text,
                generation_config={
                    "voice": voice if voice != "default" else None,
                },
                stream=False,
            )
            
            # Get audio bytes
            audio_bytes = response.candidates[0].content.parts[0].audio.bytes
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Load with pydub
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Adjust speed by modifying frame rate
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed)
                })
                
                # Export back to bytes
                output = io.BytesIO()
                audio.export(output, format="mp3")
                result = output.getvalue()
            else:
                result = audio_bytes
            
            # Cache the result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            raise TTSClientError(f"Failed to synthesize speech with Gemini TTS: {str(e)}")

    async def synthesize(self, text: str, voice: str = "default", speed: float = 1.0) -> bytes:
        """Synthesize speech using the configured provider."""
        if settings.TTS_PROVIDER == "gtts":
            return await self.synthesize_gtts(text, voice, speed)
        elif settings.TTS_PROVIDER == "gemini-tts":
            return await self.gemini_tts(text, voice, speed)
        else:
            raise TTSClientError(f"Unsupported TTS provider: {settings.TTS_PROVIDER}")
