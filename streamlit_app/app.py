import os
import uuid
import base64
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import random

# Import standard components
from components import render_card, render_audio_player, show_progress_step, render_stock_info, render_market_chart, render_tabs

# Import advanced components
from advanced_components import (
    render_provider_selector, render_correlation_matrix, render_risk_metrics_radar,
    render_portfolio_exposures, render_price_changes, render_volatility_comparison,
    render_audio_recorder, render_model_selector, render_analysis_dashboard
)

# Import utilities
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
        st.markdown("<p class='subtitle'>Advanced multi-agent financial briefing platform</p>", unsafe_allow_html=True)
    with header_col3:
        # Theme toggle button in the header
        theme_icon = "moon" if st.session_state.theme == "light" else "sun"
        st.button(f"🔄 Theme", on_click=toggle_theme)

    # Dashboard view with tabs for different views
    dashboard_tabs = st.tabs(["Market Overview", "Portfolio Analysis", "Financial Assistant"])
    
    # Tab 1: Market Overview
    with dashboard_tabs[0]:
        display_dashboard()
    
    # Tab 2: Portfolio Analysis (if we have analysis data)
    with dashboard_tabs[1]:
        if "last_analysis" in st.session_state and st.session_state.last_analysis:
            render_analysis_dashboard(st.session_state.last_analysis)
        else:
            st.info("No portfolio analysis data available yet. Use the Financial Assistant to analyze a portfolio.")
            
            # Sample portfolio for demonstration
            if st.button("Generate Sample Portfolio Analysis"):
                with st.spinner("Generating sample portfolio analysis..."):
                    # Generate mock data
                    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                    
                    # Current prices
                    prices = {sym: random.uniform(100, 500) for sym in symbols}
                    
                    # Generate historical data for the past 30 days
                    historical = {}
                    for sym in symbols:
                        base_price = prices[sym]
                        hist_data = []
                        for i in range(30):
                            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                            # Generate price with some randomness around a trend
                            close = base_price * (1 + 0.001 * i + random.uniform(-0.02, 0.02))
                            hist_data.append({"date": date, "close": close})
                        historical[sym] = hist_data
                    
                    # Call analysis agent
                    try:
                        params = {
                            "prices": prices,
                            "historical": historical,
                            "provider": "advanced",
                            "include_correlations": True,
                            "include_risk_metrics": True
                        }
                        
                        response = call_orchestrator(
                            "Analyze this portfolio", 
                            "text", 
                            params
                        )
                        
                        if response and "steps" in response:
                            for step in response["steps"]:
                                if step["tool"] == "analysis_agent":
                                    st.session_state.last_analysis = step["response"]
                                    st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to generate sample analysis: {str(e)}")
    
    # Tab 3: Financial Assistant
    with dashboard_tabs[2]:
        st.header("Financial Intelligence Assistant")
        
        # Mode selector: Text or Voice
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            text_mode = st.button(
                "💬 Text Mode", 
                help="Interact with FinAI using text",
                use_container_width=True,
                key="text_mode_button"
            )
            if text_mode:
                st.session_state.mode = "text"
        
        with mode_col2:
            voice_mode = st.button(
                "🎙️ Voice Mode", 
                help="Interact with FinAI using voice",
                use_container_width=True,
                key="voice_mode_button"
            )
            if voice_mode:
                st.session_state.mode = "voice"
        
        st.markdown(f"<div class='mode-indicator'>Current Mode: <b>{st.session_state.mode.upper()}</b></div>", unsafe_allow_html=True)
        
        # Main input area
        if st.session_state.mode == "text":
            user_input = st.text_area("Ask about market trends, specific stocks, or request a financial brief", height=100)
            audio_bytes = None
        else:  # Voice mode
            user_input = st.text_area("Your transcribed input will appear here", height=50)
            audio_bytes = render_audio_recorder()
    
    # Sidebar configuration with advanced settings
    with st.sidebar:
        st.header("Advanced Settings")
        
        # Get model selections
        model_selections = render_model_selector()
        
        # Store selections in session state
        st.session_state.model_selections = model_selections
        
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
                "Generate a market brief on tech stocks"
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
            "session_id": st.session_state.session_id
        }
        
        # Add relevant model selections from the sidebar
        if "model_selections" in st.session_state:
            # Add LLM parameters
            llm_settings = st.session_state.model_selections.get("llm", {})
            params["model"] = llm_settings.get("model")
            params["temperature"] = llm_settings.get("temperature")
            params["max_tokens"] = llm_settings.get("max_tokens")
            
            # Add voice parameters if in voice mode
            if st.session_state.mode == "voice":
                voice_settings = st.session_state.model_selections.get("voice", {})
                params["stt_provider"] = voice_settings.get("stt_provider")
                params["tts_provider"] = voice_settings.get("tts_provider")
                params["voice"] = voice_settings.get("voice")
                params["speaking_rate"] = voice_settings.get("speaking_rate")
                params["pitch"] = voice_settings.get("pitch")
            
            # Add analysis parameters
            analysis_settings = st.session_state.model_selections.get("analysis", {})
            params["analysis_provider"] = analysis_settings.get("provider")
            params["include_correlations"] = analysis_settings.get("include_correlations")
            params["include_risk_metrics"] = analysis_settings.get("include_risk_metrics")
        
        # Add audio bytes if in voice mode
        if st.session_state.mode == "voice" and st.session_state.audio_bytes:
            import base64
            audio_base64 = base64.b64encode(st.session_state.audio_bytes).decode("utf-8")
            params["audio_bytes"] = audio_base64
        
        # Add query parameters
        params["symbols"] = ",".join(st.session_state.tickers)
        params["topic"] = "finance, stocks" # Default topic
        params["limit"] = news_limit
        params["k"] = retrieve_k
        
        # Check if user input contains specific stock references
        for ticker in st.session_state.tickers:
            if ticker.lower() in user_input.lower():
                params["symbols"] = ticker
                break
                
        # Check if user wants a market brief
        if "brief" in user_input.lower() or "summary" in user_input.lower():
            params["topic"] = "market summary, finance news"
            params["limit"] = 5  # Get more articles for comprehensive brief
        
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
            try:
                response = call_orchestrator(user_input, st.session_state.mode, params)
                progress_bar.progress(100)
                
                if response:
                    output_text = response.get("output", "Sorry, I couldn't generate a response.")
                    steps = response.get("steps", [])
                    errors = response.get("errors", [])
                    audio_url = response.get("audio_url")
                    
                    # Store response in session state
                    st.session_state.last_response = response
                    
                    # Save any analysis results for the portfolio dashboard
                    for step in steps:
                        if step.get("tool") == "analysis_agent" and "response" in step:
                            st.session_state.last_analysis = step["response"]
                    
                    # Display output
                    st.header("Your Financial Brief")
                    render_card("Financial Intelligence Brief", output_text, icon="chart-line", card_type="info")
                    
                    # Play audio if available and in voice mode
                    if st.session_state.mode == "voice" and audio_url:
                        render_audio_player(audio_url, autoplay=True)
                        
                    # Add tabs for details
                    result_tabs = st.tabs(["Agent Steps", "Raw Data", "Errors"])
                    
                    # Agent Steps tab
                    with result_tabs[0]:
                        if steps:
                            st.subheader("Process Details")
                            for i, step in enumerate(steps):
                                tool_name = step.get("tool", f"Step {i+1}")
                                latency = step.get("latency_ms", 0)
                                st.markdown(f"**{tool_name}** - {latency}ms")
                                # Show simplified response
                                if "response" in step:
                                    if isinstance(step["response"], dict):
                                        # Show compact view for dict responses
                                        if "text" in step["response"]:
                                            st.markdown(f"*Response:* {step['response']['text'][:200]}...")
                                        elif "summary" in step["response"]:
                                            st.markdown(f"*Summary:* {step['response']['summary'][:200]}...")
                                    elif isinstance(step["response"], str):
                                        st.markdown(f"*Response:* {step['response'][:200]}...")
                        else:
                            st.info("No process steps recorded.")
                            
                    # Raw Data tab
                    with result_tabs[1]:
                        st.json(response)
                        
                    # Errors tab
                    with result_tabs[2]:
                        if errors:
                            st.error("The following errors occurred:")
                            for error in errors:
                                st.error(f"**{error.get('tool', 'Unknown')}**: {error.get('message', 'Unknown error')}")
                        else:
                            st.success("No errors were reported!")
                    
                    # Add to history
                    st.session_state.history.append({
                        "query": user_input,
                        "response": output_text,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "params": params
                    })
                    
                    # Display download options
                    st.subheader("Export Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Download as PDF
                        pdf_bytes = generate_pdf(output_text, "financial_brief.pdf")
                        st.download_button(
                            label="Download as PDF",
                            data=pdf_bytes,
                            file_name=f"finai_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        # Download as text
                        st.download_button(
                            label="Download as Text",
                            data=output_text,
                            file_name=f"finai_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    with col3:
                        if audio_url:
                            st.markdown(f"[Download Audio]({audio_url})")
                else:
                    st.error("Failed to get a response from the orchestrator.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
            
            # This section is now handled in the try/except block above

    # Display history
    if st.session_state.history:
        with st.container():
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
