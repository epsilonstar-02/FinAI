import os
import streamlit as st
from dotenv import load_dotenv

from components import render_card, render_audio_player, show_progress_step
from utils import call_orchestrator, call_stt, call_tts, generate_pdf

# Load environment variables
load_dotenv()
ORCH_URL = os.getenv("ORCH_URL")
VOICE_URL = os.getenv("VOICE_URL")

# Page configuration
st.set_page_config(
    page_title="FinAI - Financial Intelligence Assistant",
    page_icon="💼",
    layout="wide",
)

# Custom CSS
with open("streamlit_app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "text"
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None


def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"


def main():
    initialize_session_state()

    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("streamlit_app/assets/logo.png", width=80)
    with col2:
        st.title("FinAI - Financial Intelligence Assistant")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # Input mode selection
        mode = st.radio("Input Mode", ["text", "voice"], key="mode")
        
        # Parameters
        st.subheader("Parameters")
        news_limit = st.slider("News Article Limit", 1, 10, 3)
        retrieve_k = st.slider("Retrieval Top-k", 1, 10, 5)
        include_analysis = st.toggle("Include Market Analysis", value=True)
        
        # Theme toggle
        st.button("Toggle Theme", on_click=toggle_theme)

    # Main interface
    user_input = ""
    
    if mode == "text":
        user_input = st.text_area("Enter your financial query:", height=150)
    else:  # voice mode
        from streamlit_audio_recorder import audio_recorder
        
        st.subheader("Record your query")
        audio_bytes = audio_recorder("Click to record", "Click to stop")
        
        if audio_bytes is not None and audio_bytes != st.session_state.audio_bytes and len(audio_bytes) > 0:
            st.session_state.audio_bytes = audio_bytes
            with st.spinner("Transcribing audio..."):
                transcript = call_stt(audio_bytes)
                user_input = transcript.get("text", "")
                st.info(f"Transcript: {user_input}")

    # Process the query
    if st.button("Get Brief") and user_input:
        progress_container = st.empty()
        progress_bar = st.progress(0)
        
        # Create request parameters
        params = {
            "input": user_input,
            "mode": mode,
            "news_limit": news_limit,
            "retrieve_k": retrieve_k,
            "include_analysis": include_analysis
        }
        
        # Show step progress
        with progress_container:
            show_progress_step("Processing query", 500)
            progress_bar.progress(25)
            
            show_progress_step("Retrieving news articles", 1000)
            progress_bar.progress(50)
            
            show_progress_step("Analyzing financial data", 1500)
            progress_bar.progress(75)

        # Call orchestrator
        with st.spinner("Generating your financial brief..."):
            response = call_orchestrator(user_input, params)
            progress_bar.progress(100)
            
        # Process response
        if response and "output" in response:
            output_text = response["output"]
            
            # Display output
            st.subheader("Your Financial Brief")
            render_card("Financial Brief", output_text)
            
            # Add to history
            st.session_state.history.append({
                "query": user_input,
                "response": output_text,
                "params": params
            })
            
            # Generate voice if in voice mode
            if mode == "voice":
                with st.spinner("Generating audio response..."):
                    audio_response = call_tts(output_text, {})
                    if audio_response:
                        render_audio_player(audio_response)
            
            # Download as PDF option
            if st.button("Download as PDF", key="download_pdf"):
                pdf_bytes = generate_pdf(output_text, "financial_brief.pdf")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="financial_brief.pdf",
                    mime="application/pdf",
                )
        else:
            st.error("Failed to generate brief. Please try again.")

    # Display history
    if st.session_state.history:
        st.subheader("Recent Queries")
        for i, item in enumerate(reversed(st.session_state.history[-3:])):
            with st.expander(f"Query {len(st.session_state.history) - i}: {item['query'][:50]}..."):
                st.markdown(item["response"])


if __name__ == "__main__":
    main()
