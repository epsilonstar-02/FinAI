import os
import uuid
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from components import render_card, render_audio_player, show_progress_step, render_stock_info, render_market_chart, render_tabs
from utils import call_orchestrator, call_stt, call_tts, generate_pdf

# Load environment variables
load_dotenv()
ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8004")
VOICE_URL = os.getenv("VOICE_URL", "http://voice_agent:8006")

# Page configuration
st.set_page_config(
    page_title="FinAI - Financial Intelligence Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open("streamlit_app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "text"
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "tickers" not in st.session_state:
        st.session_state.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    if "last_response" not in st.session_state:
        st.session_state.last_response = None


def toggle_theme():
    """Toggle between light and dark theme"""
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    
    # Apply theme to body
    st.markdown(f"<script>document.body.setAttribute('data-theme', '{st.session_state.theme}');</script>", unsafe_allow_html=True)


def display_dashboard():
    """Display a simple financial dashboard with stock data"""
    # Mock data for demonstration purposes
    st.subheader("Market Overview")
    
    # Stock cards in a row
    cols = st.columns(len(st.session_state.tickers[:5]))
    
    # Sample stock data (in a real app, this would come from the API agent)
    for i, ticker in enumerate(st.session_state.tickers[:5]):
        with cols[i]:
            # Mock price and change
            import random
            price = random.uniform(100, 500)
            change = random.uniform(-5, 5)
            render_stock_info(ticker, price, change)
    
    # Market chart
    st.subheader("Market Trends")
    
    # Generate mock data for chart
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
    
    # Sample data for chart (in a real app, this would come from the API agent)
    chart_data = pd.DataFrame({
        'date': dates,
        'price': [round(100 + i * 0.5 + random.uniform(-5, 5), 2) for i in range(30)]
    })
    
    render_market_chart(chart_data)


def main():
    """Main application function"""
    initialize_session_state()

    # Apply theme
    st.markdown(f"<script>document.body.setAttribute('data-theme', '{st.session_state.theme}');</script>", unsafe_allow_html=True)

    # Header with branded layout
    header_col1, header_col2, header_col3 = st.columns([1, 4, 1])
    with header_col1:
        st.image("streamlit_app/assets/logo.png", width=80)
    with header_col2:
        st.title("FinAI - Financial Intelligence Assistant")
        st.markdown("<p class='subtitle'>Real-time market insights powered by AI</p>", unsafe_allow_html=True)
    with header_col3:
        # Theme toggle button in the header
        theme_icon = "moon" if st.session_state.theme == "light" else "sun"
        st.button(f"🔄 Theme", on_click=toggle_theme)

    # Dashboard view
    display_dashboard()
    
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)

    # Main interaction area
    st.header("Ask FinAI") 

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        
        # Input mode selection with icons
        st.subheader("💬 Input Mode")
        mode_options = ["Text", "Voice"]
        
        # Use a different key for the radio button to avoid conflicts
        selected_mode = st.radio(
            "Select input mode", 
            options=mode_options, 
            index=0 if st.session_state.mode == "text" else 1, 
            horizontal=True,
            key="input_mode_selector",
            label_visibility="collapsed"
        )
        
        # Update the session state
        st.session_state.mode = selected_mode.lower()
        
        # Query parameters
        st.subheader("⚙️ Query Parameters")
        with st.expander("Advanced Options", expanded=True):
            # Tickers multi-select
            selected_tickers = st.multiselect(
                "Ticker Symbols",
                options=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"],
                default=st.session_state.tickers[:3]
            )
            if selected_tickers:
                st.session_state.tickers = selected_tickers
                
            # Other parameters
            news_limit = st.slider("News Article Limit", 1, 10, 3)
            retrieve_k = st.slider("Retrieval Top-k", 1, 10, 5)
            include_analysis = st.toggle("Include Market Analysis", value=True)
        
        # Voice settings
        if st.session_state.mode == "voice":
            st.subheader("🎙️ Voice Settings")
            with st.expander("Voice Options"):
                voice_type = st.selectbox(
                    "Voice Type",
                    options=["en-US-Neural2-F", "en-US-Neural2-M", "en-GB-Neural2-F", "en-GB-Neural2-M"],
                    index=0
                )
                speaking_rate = st.slider("Speaking Rate", 0.5, 1.5, 1.0, 0.1)
                pitch = st.slider("Pitch", -5.0, 5.0, 0.0, 0.5)
                
        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About FinAI")
        st.sidebar.info(
            "FinAI is an advanced multi-agent financial intelligence system "
            "designed to provide real-time market insights through natural language interaction."
        )

    # Main interaction area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input area based on mode
        if st.session_state.mode == "text":
            user_input = st.text_area(
                "Enter your financial query:", 
                height=100,
                placeholder="E.g., What's the current market outlook for tech stocks?"
            )
            
            # Quick question suggestions
            st.markdown("<p><strong>Try asking:</strong></p>", unsafe_allow_html=True)
            suggestion_cols = st.columns(3)
            suggestions = [
                "What's the current price of AAPL?",
                "How has the market performed this week?",
                "What are the latest news about Tesla?"
            ]
            
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i]:
                    if st.button(f"{suggestion}", key=f"suggestion_{i}"):
                        user_input = suggestion
                        st.rerun()
                        
        else:  # voice mode
            from streamlit_audio_recorder import audio_recorder
            
            st.markdown("### Record Your Query")
            audio_bytes = audio_recorder("Click to record", "Click to stop recording")
            
            user_input = ""
            if audio_bytes is not None and audio_bytes != st.session_state.audio_bytes and len(audio_bytes) > 0:
                st.session_state.audio_bytes = audio_bytes
                with st.spinner("Transcribing audio..."):
                    transcript = call_stt(audio_bytes)
                    user_input = transcript.get("text", "")
                    if user_input:
                        st.success(f"Transcript: {user_input}")
                    else:
                        st.error("Failed to transcribe audio. Please try again.")
    
    with col2:
        # Session info
        st.markdown("### Session Info")
        st.markdown(f"**Session ID:** {st.session_state.session_id[:8]}...")
        st.markdown(f"**Mode:** {st.session_state.mode.capitalize()}")
        st.markdown(f"**Queries:** {len(st.session_state.history)}")
        
        # Action buttons
        st.markdown("### Actions")
        if st.button("Clear History", key="clear_history"):
            st.session_state.history = []
            st.session_state.last_response = None
            st.success("History cleared successfully!")
            st.rerun()

    # Process button - separate from columns
    process_btn = st.button("🚀 Generate Financial Brief", key="process_btn", type="primary")
    
    # Process the query when button is clicked
    if process_btn and user_input:
        progress_container = st.empty()
        progress_bar = st.progress(0)
        
        # Create request parameters
        params = {
            "mode": st.session_state.mode,
            "tickers": st.session_state.tickers,
            "news_limit": news_limit,
            "retrieve_k": retrieve_k,
            "include_analysis": include_analysis,
            "session_id": st.session_state.session_id
        }
        
        # Add audio bytes if in voice mode
        if st.session_state.mode == "voice" and st.session_state.audio_bytes:
            params["audio_bytes"] = st.session_state.audio_bytes
            
        # Voice settings
        if st.session_state.mode == "voice" and 'voice_type' in locals():
            params["voice"] = voice_type
            params["speaking_rate"] = speaking_rate
            params["pitch"] = pitch
        
        # Show step progress
        with progress_container:
            show_progress_step("Processing query", 500)
            progress_bar.progress(20)
            
            show_progress_step("Retrieving financial data", 800)
            progress_bar.progress(40)
            
            show_progress_step("Gathering news articles", 800)
            progress_bar.progress(60)
            
            show_progress_step("Analyzing market trends", 800)
            progress_bar.progress(80)

        # Call orchestrator
        with st.spinner("Generating your financial brief..."):
            response = call_orchestrator(user_input, params)
            progress_bar.progress(100)
            
        # Process response
        if response and response.get("status") == "success" and "result" in response:
            result = response["result"]
            output_text = result.get("text", "")
            steps = response.get("steps", [])
            
            # Store response in session state
            st.session_state.last_response = response
            
            # Display output
            st.header("Your Financial Brief")
            render_card("Financial Intelligence Brief", output_text, icon="chart-line", card_type="info")
            
            # Show steps information
            if steps:
                with st.expander("Process Details", expanded=False):
                    for step in steps:
                        st.markdown(f"**{step['tool']}**: {step['latency_ms']}ms")
                        if 'response' in step and isinstance(step['response'], dict):
                            st.json(step['response'])
            
            # Add to history
            st.session_state.history.append({
                "query": user_input,
                "response": output_text,
                "params": params,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Generate voice if in voice mode
            if st.session_state.mode == "voice" and "audio" in result:
                with st.spinner("Playing audio response..."):
                    render_audio_player(result["audio"], autoplay=True)
            elif st.session_state.mode == "voice":
                # Fallback to calling TTS directly if not in response
                with st.spinner("Generating audio response..."):
                    voice_params = {
                        "voice": params.get("voice", "en-US-Neural2-F"),
                        "speaking_rate": params.get("speaking_rate", 1.0),
                        "pitch": params.get("pitch", 0.0)
                    }
                    audio_response = call_tts(output_text, voice_params)
                    if audio_response:
                        render_audio_player(audio_response, autoplay=True)
            
            # Action buttons for response
            col1, col2 = st.columns(2)
            with col1:
                # Download as PDF option
                pdf_bytes = generate_pdf(output_text, "financial_brief.pdf")
                st.download_button(
                    label="Download as PDF",
                    data=pdf_bytes,
                    file_name=f"finai_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
            with col2:
                # Share option (mock functionality)
                if st.button("Share this Brief", key="share_brief"):
                    st.success("Sharing link copied to clipboard!")
        else:
            error_msg = "Failed to generate brief. Please try again."
            if response and "errors" in response:
                error_details = "; ".join([e.get("message", "") for e in response["errors"] if "message" in e])
                if error_details:
                    error_msg += f" Details: {error_details}"
            st.error(error_msg)

    # Display history
    if st.session_state.history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Query History")
        
        # Tab options for history display
        def show_list_view():
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"**{item['timestamp']}** - {item['query'][:50]}...", expanded=(i==0)):
                    render_card("Response", item["response"], card_type="default")
        
        def show_table_view():
            history_data = [{
                "Time": item["timestamp"],
                "Query": item["query"][:50] + ("..." if len(item["query"]) > 50 else ""),
                "Mode": item["params"].get("mode", "text").capitalize()
            } for item in st.session_state.history]
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        
        # Display tabs
        render_tabs({
            "List View": show_list_view,
            "Table View": show_table_view
        })


if __name__ == "__main__":
    main()
