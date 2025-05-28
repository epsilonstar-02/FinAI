import io
import json
import requests
from fpdf import FPDF


def call_orchestrator(input_text, params):
    """
    Calls the orchestrator API with the given input and parameters.
    
    Args:
        input_text: The user's input text
        params: Dictionary of parameters for the orchestrator
        
    Returns:
        JSON response from the orchestrator or None if the request failed
    """
    try:
        import os
        import base64
        url = f"{os.getenv('ORCH_URL', 'http://orchestrator:8004')}/run"
        
        request_data = {
            "mode": params.get("mode", "text"),
            "params": {
                "query": input_text,
                "tickers": params.get("tickers", ["AAPL", "MSFT", "GOOGL"]),
                "news_limit": params.get("news_limit", 3),
                "retrieve_k": params.get("retrieve_k", 5),
                "include_analysis": params.get("include_analysis", True)
            },
            "session_id": params.get("session_id", "streamlit-session")
        }
        
        # If in voice mode and we have audio bytes, include them
        if params.get("mode") == "voice" and params.get("audio_bytes"):
            # Encode audio bytes to base64
            audio_base64 = base64.b64encode(params["audio_bytes"]).decode("utf-8")
            request_data["params"]["audio_bytes"] = audio_base64
            
        response = requests.post(
            url,
            json=request_data,
            timeout=60  # Increased timeout for more complex operations
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling orchestrator: {e}")
        return None


def call_stt(file_bytes):
    """
    Calls the Speech-to-Text API with the given audio file bytes.
    
    Args:
        file_bytes: Bytes of the audio file
        
    Returns:
        JSON response containing the transcribed text or an empty dict if failed
    """
    try:
        import os
        import base64
        
        # Get the voice agent URL from environment variables
        url = f"{os.getenv('VOICE_URL', 'http://voice_agent:8006')}/stt"
        
        # Convert audio bytes to base64 for JSON serialization
        audio_base64 = base64.b64encode(file_bytes).decode("utf-8")
        
        # Send as JSON payload instead of file upload
        response = requests.post(
            url,
            json={"audio": audio_base64},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling STT service: {e}")
        return {"text": ""}


def call_tts(text, params):
    """
    Calls the Text-to-Speech API with the given text.
    
    Args:
        text: Text to convert to speech
        params: Optional parameters for the TTS service
        
    Returns:
        Audio bytes or None if the request failed
    """
    try:
        import os
        import base64
        
        # Get the voice agent URL from environment variables
        url = f"{os.getenv('VOICE_URL', 'http://voice_agent:8006')}/tts"
        
        # Prepare the request data
        request_data = {
            "text": text,
            "voice": params.get("voice", "en-US-Neural2-F"),  # Default voice
            "speaking_rate": params.get("speaking_rate", 1.0),  # Normal speed
            "pitch": params.get("pitch", 0.0)  # Default pitch
        }
        
        # Send the request
        response = requests.post(
            url,
            json=request_data,
            timeout=45  # Longer timeout for TTS generation
        )
        response.raise_for_status()
        
        # If the response contains base64 audio, decode it
        if 'audio' in response.json():
            audio_base64 = response.json()['audio']
            return base64.b64decode(audio_base64)
        
        # Fallback to raw content
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error calling TTS service: {e}")
        return None


def generate_pdf(text, filename):
    """
    Generates a PDF document with the given text.
    
    Args:
        text: The text content for the PDF
        filename: The name of the PDF file
        
    Returns:
        PDF document as bytes
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Set up the PDF
    pdf.set_font("Helvetica", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Financial Intelligence Brief", ln=True, align="C")
    pdf.ln(5)
    
    # Add content
    pdf.set_font("Helvetica", size=11)
    
    # Split text into paragraphs
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        pdf.multi_cell(0, 6, paragraph.strip())
        pdf.ln(3)
    
    # Add footer with date
    import datetime
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_y(-15)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated by FinAI on {current_date}", 0, 0, "C")
    
    # Get the PDF as a string and encode to bytes
    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin1")
    return pdf_output
