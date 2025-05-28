import io
import json # Keep for potential future use, though not strictly needed now if orchestrator handles all
import requests
from fpdf import FPDF
import os # For environment variables
import base64 # For audio encoding/decoding


def call_orchestrator(input_text, params: dict):
    """
    Calls the orchestrator API with the given input and parameters.
    
    Args:
        input_text: The user's input text
        params: Dictionary of parameters for the orchestrator, including mode, session_id,
                model settings, voice settings (if applicable), analysis settings,
                tickers, news_limit, retrieve_k, audio_bytes_b64 (if applicable).
        
    Returns:
        JSON response from the orchestrator or None if the request failed
    """
    try:
        url = f"{os.getenv('ORCH_URL', 'http://orchestrator:8004')}/run"
        
        # The orchestrator expects a specific payload structure.
        # Based on original app.py, it seems 'params' from app.py is the main payload
        # and 'input_text' is part of it.
        # Let's assume the orchestrator expects `query` at the top level,
        # and other settings within a `config` or `settings` sub-dictionary.
        # For now, constructing as per the original implied structure.

        request_data = {
            "query": input_text, # User's main query text
            "mode": params.get("mode", "text"),
            "session_id": params.get("session_id", "streamlit-session"),
            # Nested dictionary for all other configuration parameters
            "config_params": {
                "model": params.get("model"),
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                
                "stt_provider": params.get("stt_provider"),
                "tts_provider": params.get("tts_provider"),
                "voice": params.get("voice"),
                "speaking_rate": params.get("speaking_rate"),
                "pitch": params.get("pitch"),
                
                "analysis_provider": params.get("analysis_provider"),
                "include_correlations": params.get("include_correlations"),
                "include_risk_metrics": params.get("include_risk_metrics"),
                
                "tickers": params.get("tickers"),
                "news_limit": params.get("news_limit"),
                "retrieve_k": params.get("retrieve_k"),
                "topic": params.get("topic"),
                "symbols_mentioned": params.get("symbols_mentioned"), # Specific to query
                "analysis_params": params.get("analysis_params") # For sample analysis
            }
        }
        
        if params.get("mode") == "voice" and params.get("audio_bytes_b64"):
            request_data["audio_bytes_b64"] = params["audio_bytes_b64"]
            
        response = requests.post(
            url,
            json=request_data,
            timeout=120  # Increased timeout for potentially long operations
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling orchestrator: {e}") # Log to console
        # Optionally, return a dict with error info for UI display
        return {"error": True, "message": str(e), "output": "Failed to connect to the financial intelligence engine."}
    except Exception as ex: # Catch other potential errors during request prep
        print(f"Unexpected error in call_orchestrator: {ex}")
        return {"error": True, "message": str(ex), "output": "An unexpected error occurred while preparing the request."}


def call_stt(file_bytes):
    """
    Calls the Speech-to-Text API.
    """
    try:
        url = f"{os.getenv('VOICE_URL', 'http://voice_agent:8006')}/stt"
        audio_base64 = base64.b64encode(file_bytes).decode("utf-8")
        
        response = requests.post(url, json={"audio_base64": audio_base64}, timeout=30) # Key changed for clarity
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling STT service: {e}")
        return {"text": "", "error": str(e)} # Return error info


def call_tts(text, params: dict):
    """
    Calls the Text-to-Speech API.
    Returns:
        Base64 encoded audio string or None if failed.
    """
    try:
        url = f"{os.getenv('VOICE_URL', 'http://voice_agent:8006')}/tts"
        request_data = {
            "text": text,
            "voice": params.get("voice", "en-US-Neural2-F"),
            "speaking_rate": params.get("speaking_rate", 1.0),
            "pitch": params.get("pitch", 0.0)
        }
        
        response = requests.post(url, json=request_data, timeout=45)
        response.raise_for_status()
        
        # Expecting JSON response with base64 audio, e.g., {"audio_base64": "..."}
        data = response.json()
        if 'audio_base64' in data:
            return data['audio_base64'] # Return base64 string directly
        else:
            print("TTS service did not return audio_base64.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling TTS service: {e}")
        return None
    except json.JSONDecodeError:
        print(f"TTS service returned non-JSON response: {response.text}")
        return None


def generate_pdf(text, filename): # Filename argument not used if returning bytes
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FinAI - Financial Intelligence Brief", ln=True, align="C")
    pdf.ln(8) # More space after title
    
    pdf.set_font("Helvetica", size=11)
    
    # Enhanced paragraph handling: split by '\n' and handle lines carefully
    # This is a simplified version. For complex markdown, a library would be better.
    lines = text.split('\n')
    for line in lines:
        if line.startswith("### "): # H3
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, line.replace("### ", "").strip())
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("## "): # H2
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, line.replace("## ", "").strip())
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("# "): # H1
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 9, line.replace("# ", "").strip())
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("* ") or line.startswith("- "): # Bullet points
            pdf.cell(5) # Indent for bullet
            pdf.multi_cell(0, 6, f"• {line[2:].strip()}")
        elif line.strip() == "---": # Horizontal line
            pdf.line(pdf.get_x(), pdf.get_y() + 2, pdf.get_x() + pdf.w - 2 * pdf.l_margin, pdf.get_y() + 2)
            pdf.ln(4)
        else:
            pdf.multi_cell(0, 6, line.strip())
        pdf.ln(1) # Small space between multi_cell blocks or lines
    
    import datetime
    pdf.set_y(-15) # Position for footer
    pdf.set_font("Helvetica", "I", 8)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated by FinAI on {current_date}", 0, 0, "C")
    
    pdf_output_bytes = pdf.output(dest="S").encode("latin1") # FPDF returns string, encode to bytes
    return pdf_output_bytes