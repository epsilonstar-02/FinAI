import google.generativeai as genai
from .config import settings
import asyncio
from typing import Dict, Any


class LLMClientError(Exception):
    """Exception raised for errors in the LLM client."""
    pass


# Configure the Google Generative AI client
genai.configure(api_key=settings.GEMINI_API_KEY)


async def generate_text(prompt: str) -> str:
    """
    Generate text using Google Gemini model.
    
    Args:
        prompt: The prompt to send to the model
        
    Returns:
        The generated text response
        
    Raises:
        LLMClientError: If there's an error in generating text
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Create a chat session
        chat = model.start_chat()
        
        # Send the message and get response with timeout
        response = await asyncio.wait_for(
            chat.send_message_async(prompt, 
                                  generation_config={
                                      "temperature": 0.2,
                                      "top_k": 40,
                                      "top_p": 0.95,
                                      "max_output_tokens": 1024,
                                  }),
            timeout=settings.TIMEOUT
        )
        
        # Return the text content
        return response.text
    except asyncio.TimeoutError:
        raise LLMClientError(f"Request timed out after {settings.TIMEOUT} seconds")
    except Exception as e:
        raise LLMClientError(f"Error generating text: {str(e)}")
