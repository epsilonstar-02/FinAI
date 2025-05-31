# streamlit_app/utils.py
# Refactoring call_orchestrator to align with Orchestrator's RunRequest.
# Reviewing call_stt. Removing call_tts as Orchestrator handles it.

import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry # Corrected import for Retry from urllib3
from fpdf import FPDF
import os
import base64
import streamlit as st # For logging/displaying errors within Streamlit context
from typing import Dict, Any, Optional # Added Optional

# Environment variables for URLs
ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8004")
VOICE_URL = os.getenv("VOICE_URL", "http://voice_agent:8006")

# Adjust for local development if needed (as in original app.py)
if os.getenv("STREAMLIT_ENV", "prod").lower() == "dev":
    if ORCH_URL == "http://orchestrator:8004": ORCH_URL = "http://localhost:8004"
    if VOICE_URL == "http://voice_agent:8006": VOICE_URL = "http://localhost:8006"


def create_retry_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504, 408, 429)):
    """Create a requests session with retry functionality."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"] # Allow retries for POST etc.
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global session for utils module
_http_session = create_retry_session()


def call_orchestrator(input_text: str, mode: str, orchestrator_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calls the orchestrator API with input text, mode, and a dictionary of parameters.
    
    Args:
        input_text: The user's primary input query.
        mode: The interaction mode ("text" or "voice").
        orchestrator_params: A dictionary containing all other parameters that the
                             Orchestrator's RunRequest.params field expects. This includes
                             LLM settings, STT/TTS settings (if mode is voice and audio is passed),
                             analysis settings, tickers, news_limit, retrieve_k, audio_bytes_b64, etc.
        
    Returns:
        JSON response from the orchestrator or a dict with error info if the request failed.
    """
    endpoint_url = f"{ORCH_URL}/run"
    
    # Construct the payload according to Orchestrator's RunRequest model
    payload = {
        "input": input_text,
        "mode": mode,
        "params": orchestrator_params # All other settings go into this nested dict
    }
    
    st.write("Orchestrator Payload:", payload) # For debugging

    try:
        response = _http_session.post(
            endpoint_url,
            json=payload,
            timeout=180  # Increased timeout for potentially long orchestrations
        )
        response.raise_for_status() # Raises HTTPError for 4xx/5xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        err_msg = f"Orchestrator HTTP error: {http_err.response.status_code} - {http_err.response.text}"
        st.error(err_msg)
        print(err_msg) # Also print to console for server-side logs
        return {"error": True, "message": str(http_err), "status_code": http_err.response.status_code, "details": http_err.response.text, "output": "Failed to get a response from the financial intelligence engine due to a server error."}
    except requests.exceptions.RequestException as req_err:
        err_msg = f"Orchestrator connection error: {req_err}"
        st.error(err_msg)
        print(err_msg)
        return {"error": True, "message": str(req_err), "output": "Failed to connect to the financial intelligence engine."}
    except Exception as ex:
        err_msg = f"Unexpected error calling orchestrator: {ex}"
        st.error(err_msg)
        print(err_msg)
        return {"error": True, "message": str(ex), "output": "An unexpected error occurred while communicating with the financial intelligence engine."}


def call_stt(audio_file_bytes: bytes, stt_provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Calls the Voice Agent's Speech-to-Text API.
    The Voice Agent's /stt endpoint expects multipart/form-data with a 'file'.
    """
    endpoint_url = f"{VOICE_URL}/stt"
    files = {'file': ('audio.wav', audio_file_bytes, 'audio/wav')}
    params = {}
    if stt_provider:
        params['provider'] = stt_provider

    try:
        # Use a non-retrying session for STT if it's quick, or use _http_session
        response = requests.post(endpoint_url, files=files, params=params, timeout=45) # Adjust timeout as needed for STT
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        err_msg = f"STT Voice Agent HTTP error: {http_err.response.status_code} - {http_err.response.text}"
        st.error(err_msg)
        print(err_msg)
        return {"text": "", "error": str(http_err), "message": err_msg, "details": http_err.response.text}
    except requests.exceptions.RequestException as e:
        err_msg = f"Error calling STT service: {e}"
        st.error(err_msg)
        print(err_msg)
        return {"text": "", "error": str(e), "message": err_msg}


# call_tts is removed as the Orchestrator should handle TTS and return audio_output_b64
# The Streamlit app will use the audio_output_b64 from the Orchestrator's response.


def generate_pdf(text_content: str, default_filename: str = "financial_brief.pdf") -> bytes:
    """
    Generates a PDF document from the given text content.
    
    Args:
        text_content: The string content to put into the PDF.
        default_filename: Not used by function return, but kept for signature consistency.
        
    Returns:
        PDF content as bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12) # Using Helvetica, a standard PDF font
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FinAI - Financial Intelligence Brief", ln=True, align="C")
    pdf.ln(10) 
    
    pdf.set_font("Helvetica", "", 11) # Reset to normal font for content
    
    # Simple paragraph handling: split by double newlines for paragraphs,
    # and handle single newlines within those paragraphs as line breaks.
    paragraphs = text_content.split('\n\n')
    
    for para_text in paragraphs:
        if not para_text.strip():
            continue

        # Handle potential markdown-like headers (simple version)
        if para_text.startswith("### "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, para_text.replace("### ", "").strip())
            pdf.set_font("Helvetica", "", 11)
        elif para_text.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, para_text.replace("## ", "").strip())
            pdf.set_font("Helvetica", "", 11)
        elif para_text.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 9, para_text.replace("# ", "").strip())
            pdf.set_font("Helvetica", "", 11)
        elif para_text.startswith("* ") or para_text.startswith("- "): # Simple bullet points
            # Split into bullet lines
            bullet_lines = para_text.split('\n')
            for b_line in bullet_lines:
                if b_line.strip().startswith(("* ", "- ")):
                    pdf.cell(5) # Indent
                    pdf.multi_cell(0, 6, f"• {b_line.strip()[2:]}") # Replace markdown bullet with PDF bullet
                else: # Continuation of a bullet or normal line within "paragraph"
                    pdf.multi_cell(0, 6, b_line.strip())
        elif para_text.strip() == "---":
            pdf.line(pdf.get_x(), pdf.get_y() + 2, pdf.get_x() + pdf.w - 2 * pdf.l_margin, pdf.get_y() + 2)
            pdf.ln(4)
        else:
            pdf.multi_cell(0, 6, para_text.strip()) # Line height 6
        pdf.ln(2) # Small space after each processed paragraph block
    
    # Footer
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated by FinAI on {current_datetime_str}", 0, 0, "C")
    
    # FPDF output method 'S' returns the document as a string.
    # For bytes, especially with Python 3, encode it. latin1 is common for FPDF.
    pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_output_bytes