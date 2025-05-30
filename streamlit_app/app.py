import os
import uuid
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import random
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

from components import render_card, render_audio_player, show_progress_step, render_stock_info, render_market_chart, render_tabs
from advanced_components import (
    render_provider_selector,
    render_model_selector, render_analysis_dashboard
)
from utils import call_orchestrator, call_stt, call_tts, generate_pdf

SAMPLE_RATE = 16000
CHANNELS = 1

if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []
if "audio_bytes_to_process" not in st.session_state:
    st.session_state.audio_bytes_to_process = None
if "webrtc_is_playing" not in st.session_state:
    st.session_state.webrtc_is_playing = False

class AudioRecorderProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self._frames_buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self._frames_buffer.append(frame.to_ndarray())
        return frame

    def get_recorded_data_and_clear(self) -> bytes | None:
        if not self._frames_buffer:
            return None
        
        all_samples = np.concatenate(self._frames_buffer, axis=1)

        if all_samples.ndim > 1 and all_samples.shape[0] > 1:
            all_samples = all_samples[0, :]
        all_samples = all_samples.flatten()

        if np.issubdtype(all_samples.dtype, np.floating):
            all_samples = (all_samples * 32767).astype(np.int16)
        elif all_samples.dtype != np.int16:
            all_samples = all_samples.astype(np.int16)
            
        audio_bytes = all_samples.tobytes()
        self._frames_buffer.clear()
        return audio_bytes

load_dotenv()
# Default URLs for containerized environment
ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8004")
VOICE_URL = os.getenv("VOICE_URL", "http://voice_agent:8006")

# Fallback to localhost for local development if needed
if ORCH_URL == "http://orchestrator:8004" and os.getenv("ENVIRONMENT", "prod").lower() == "dev":
    ORCH_URL = "http://localhost:8004"
if VOICE_URL == "http://voice_agent:8006" and os.getenv("ENVIRONMENT", "prod").lower() == "dev":
    VOICE_URL = "http://localhost:8006"

st.set_page_config(
    page_title="FinAI - Financial Intelligence Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    with open("streamlit_app/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("styles.css not found. The app may not be styled correctly.")


def initialize_session_state():
    defaults = {
        "history": [],
        "mode": "text",
        "theme": "light",
        "audio_bytes": None,
        "session_id": str(uuid.uuid4()),
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "last_response": None,
        "last_analysis": None,
        "user_input_text": "",
        "transcript_text": "",
        "model_selections": {} 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def toggle_theme_js():
    theme_js = """
        <script>
            function toggleTheme() {
                const currentTheme = document.body.getAttribute('data-theme');
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                document.body.setAttribute('data-theme', newTheme);
            }
            document.body.setAttribute('data-theme', '%s');
        </script>
    """ % st.session_state.get("theme", "light")
    st.markdown(theme_js, unsafe_allow_html=True)

def python_toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()

def display_dashboard():
    st.subheader("Market Overview")
    
    cols = st.columns(len(st.session_state.tickers[:5]))
    
    for i, ticker in enumerate(st.session_state.tickers[:5]):
        with cols[i]:
            price = random.uniform(100, 500)
            change = random.uniform(-5, 5)
            render_stock_info(ticker, price, change)
    
    st.subheader("Market Trends (Sample Data)")
    dates = pd.to_datetime([(datetime.now() - timedelta(days=i)) for i in range(29, -1, -1)])
    chart_data = pd.DataFrame({
        'date': dates,
        'price': [round(150 + i * 0.8 + random.uniform(-10, 10), 2) for i in range(30)]
    })
    render_market_chart(chart_data)


def main():
    initialize_session_state()
    toggle_theme_js() 

    st.markdown(f'''
    <div class="app-header">
        <img src="streamlit_app/assets/logo.png" class="app-logo" alt="FinAI Logo" /> 
        <div>
            <h1 class="app-title">FinAI</h1>
            <p class="subtitle">Advanced Multi-Agent Financial Intelligence</p>
        </div>
        <div style="margin-left: auto; display: flex; align-items: center;">
            <button onclick="toggleTheme()" title="Toggle Theme" class="theme-toggle-button">
                <i class="fas fa-adjust"></i>
            </button>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    tab_titles = ["📊 Market Overview", "📈 Portfolio Analysis", "🤖 Financial Assistant"]
    dashboard_tabs = st.tabs(tab_titles)
    
    with dashboard_tabs[0]:
        display_dashboard()
    
    with dashboard_tabs[1]:
        if st.session_state.last_analysis:
            render_analysis_dashboard(st.session_state.last_analysis)
        else:
            st.info("No portfolio analysis data. Use the 'Financial Assistant' or generate a sample below.")
            if st.button("🔬 Generate Sample Portfolio Analysis"):
                with st.spinner("Generating sample analysis... This might take a moment."):
                    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
                    prices = {sym: random.uniform(100, 1000) for sym in symbols}
                    historical = {}
                    for sym in symbols:
                        base_price = prices[sym]
                        hist_data = []
                        for i in range(60, 0, -1):
                            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                            close = base_price * (1 - 0.0005 * i + random.uniform(-0.015, 0.015))
                            hist_data.append({"date": date, "close": round(max(1, close),2) })
                        historical[sym] = hist_data
                    
                    try:
                        analysis_settings = st.session_state.model_selections.get("analysis", {})
                        params = {
                            "prices": prices, "historical": historical,
                            "provider": analysis_settings.get("provider", "advanced"),
                            "include_correlations": analysis_settings.get("include_correlations", True),
                            "include_risk_metrics": analysis_settings.get("include_risk_metrics", True)
                        }
                        response = call_orchestrator("Analyze this sample portfolio", {"analysis_params": params}) 
                        
                        if response and "steps" in response:
                            analysis_step_found = False
                            for step in response["steps"]:
                                if step.get("tool") == "analysis_agent" and "response" in step:
                                    st.session_state.last_analysis = step["response"]
                                    analysis_step_found = True
                                    break
                            if analysis_step_found:
                                st.success("Sample analysis generated!")
                                st.rerun()
                            else:
                                st.warning("Analysis agent did not return data in the expected format.")
                        else:
                            st.error("Failed to generate sample analysis or orchestrator returned an error.")
                    except Exception as e:
                        st.error(f"Error during sample analysis generation: {str(e)}")

    with dashboard_tabs[2]:
        st.header("Financial Intelligence Assistant")
        
        mode_cols = st.columns(2)
        with mode_cols[0]:
            if st.button("💬 Text Mode", use_container_width=True, type="secondary" if st.session_state.mode != "text" else "primary"):
                st.session_state.mode = "text"
                st.rerun()
        with mode_cols[1]:
            if st.button("🎙️ Voice Mode", use_container_width=True, type="secondary" if st.session_state.mode != "voice" else "primary"):
                st.session_state.mode = "voice"
                st.rerun()
        
        st.caption(f"Current Mode: **{st.session_state.mode.upper()}**")
        st.markdown("---")

        processed_user_query = ""

        if st.session_state.mode == "text":
            st.session_state.user_input_text = st.text_area(
                "Enter your financial query:",
                value=st.session_state.user_input_text,
                height=100,
                placeholder="E.g., What's the current market outlook for tech stocks?"
            )
            processed_user_query = st.session_state.user_input_text

            st.markdown("<p style='font-size: 0.9em; margin-bottom: 0.5em;'><strong>💡 Try asking:</strong></p>", unsafe_allow_html=True)
            suggestion_cols = st.columns(3)
            suggestions = [
                "Price of AAPL?",
                "Market performance this week?",
                "Tech stocks market brief"
            ]
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        st.session_state.user_input_text = suggestion
                        processed_user_query = suggestion
                        st.rerun()
        
        else: 
            st.markdown("##### Record Your Query")
            st.caption("Click 'Start' to record, then 'Stop'. Audio will be processed automatically.")

            if "audio_processor_instance" not in st.session_state:
                st.session_state.audio_processor_instance = None
            
            def audio_processor_factory():
                processor = AudioRecorderProcessor()
                st.session_state.audio_processor_instance = processor
                return processor

            col_rec, col_stop = st.columns(2)
            if col_rec.button("🎙️ Start Recording", key="start_rec_btn", use_container_width=True, disabled=st.session_state.webrtc_is_playing):
                st.session_state.webrtc_is_playing = True
                st.rerun()

            if col_stop.button("⏹️ Stop Recording", key="stop_rec_btn", use_container_width=True, disabled=not st.session_state.webrtc_is_playing):
                st.session_state.webrtc_is_playing = False
                st.rerun()

            if st.session_state.webrtc_is_playing:
                webrtc_ctx = webrtc_streamer(
                    key="audio_stream_recorder",
                    mode=WebRtcMode.SENDRECV,
                    audio_processor_factory=audio_processor_factory,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": False, "audio": True},
                    desired_playing_state=st.session_state.webrtc_is_playing,
                )
            
            if not st.session_state.webrtc_is_playing and st.session_state.get("audio_processor_instance"):
                audio_bytes_recorded = st.session_state.audio_processor_instance.get_recorded_data_and_clear()
                if audio_bytes_recorded:
                    st.session_state.audio_bytes_to_process = audio_bytes_recorded
                    st.session_state.audio_processor_instance = None
                    st.session_state.audio_bytes = audio_bytes_recorded
                    st.session_state.last_audio_bytes_processed = audio_bytes_recorded
                    st.rerun()

            if st.session_state.audio_bytes_to_process:
                with st.spinner("Transcribing audio..."):
                    st.audio(st.session_state.audio_bytes_to_process, format="audio/wav", sample_rate=SAMPLE_RATE)
                    
                    transcript_response = call_stt(st.session_state.audio_bytes_to_process)
                    st.session_state.transcript_text = transcript_response.get("text", "")
                    st.session_state.audio_bytes_to_process = None
                    
                    if st.session_state.transcript_text:
                        st.success("✅ Transcription complete!")
                        st.info(f"🗣️ Transcript: \"{st.session_state.transcript_text}\"")
                        processed_user_query = st.session_state.transcript_text
                    else:
                        st.error("Could not transcribe audio. Please try again.")
            
            if st.session_state.transcript_text:
                st.info(f"🗣️ Last Transcript: \"{st.session_state.transcript_text}\" (Ready to send)")
                processed_user_query = st.session_state.transcript_text

        if st.button("🚀 Generate Financial Brief", type="primary", use_container_width=True, disabled=not processed_user_query.strip()):
            if not processed_user_query.strip():
                st.warning("Please enter a query or record your voice.")
            else:
                st.session_state.query_to_process = processed_user_query
                st.session_state.trigger_processing = True
                st.rerun()

    # --- Updated Sidebar ---
    with st.sidebar:
        st.image("streamlit_app/assets/logo.png", width=60)
        st.header("Settings & Info")
        # Removed first st.markdown("---") for cleaner look after header

        with st.expander("🤖 Model Configuration", expanded=False): # Set to False for cleaner initial view
            # Pass show_subheader=False to avoid duplicate subheader from the function
            st.session_state.model_selections = render_model_selector(show_subheader=False) 
        
        st.markdown("---")

        st.subheader("Input Mode")
        mode_options = ["Text", "Voice"]
        selected_mode_sidebar = st.radio(
            "Select input mode", options=mode_options, # Label is effectively covered by subheader
            index=mode_options.index(st.session_state.mode.capitalize()),
            horizontal=True, key="sidebar_input_mode_selector", label_visibility="collapsed"
        )
        st.markdown("---")

        st.subheader("Query Parameters")
        with st.expander("Advanced Query Options", expanded=False): # More descriptive expander label
            all_ticker_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT", "BTC-USD", "ETH-USD"]
            # Ensure news_limit and retrieve_k are initialized in session_state if they are to be used directly from there
            # or passed around. For now, they are function-local in the processing block.
            # For robust access in orch_params, ensure they are defined before that block.
            # One way is to assign them to st.session_state here if not already.
            # However, the original code defines them as local vars in the processing block based on sliders.
            # This part of the code (sliders directly creating vars) is fine.
            
            st.session_state.tickers = st.multiselect(
                "Ticker Symbols for Context",
                options=all_ticker_options,
                default=st.session_state.tickers
            )
            # These sliders will create 'news_limit' and 'retrieve_k' in the scope of the main() function when this part of sidebar is rendered.
            # Ensure these names don't clash or are handled correctly in the processing block.
            # The current code re-reads them from session_state.model_selections or uses the slider values from the main scope,
            # which can be confusing. Best to ensure these values are stored in session_state if used across reruns or callbacks.
            # For sliders, the key directly updates st.session_state.
            # So, news_limit can be st.session_state.news_limit_slider.

            _news_limit = st.slider("News Article Limit", 1, 10, st.session_state.get("news_limit_slider", 3), key="news_limit_slider")
            _retrieve_k = st.slider("Retrieval Top-k (Documents)", 1, 10, st.session_state.get("retrieve_k_slider", 5), key="retrieve_k_slider")

        st.markdown("---")
        
        st.subheader("Session")
        st.caption(f"ID: {st.session_state.session_id[:8]}...")
        st.caption(f"Queries: {len(st.session_state.history)}")
        if st.button("Clear Chat History", key="clear_history_sidebar", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_response = None
            st.session_state.user_input_text = ""
            st.session_state.transcript_text = ""
            st.success("History cleared!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")

        st.subheader("About FinAI")
        st.info(
            "FinAI is an advanced multi-agent financial intelligence system providing real-time market insights."
        )
        st.caption(f"Version 1.0.0 | Theme: {st.session_state.theme.capitalize()}")
        
        # Removed redundant theme toggle button from sidebar
        # The header theme toggle is preferred.

    # Processing logic (triggered by the flag set above)
    # Initialize news_limit and retrieve_k here to ensure they have values
    # before being used in orch_params, especially if the sidebar part isn't rendered first.
    news_limit = st.session_state.get("news_limit_slider", 3)
    retrieve_k = st.session_state.get("retrieve_k_slider", 5)

    if st.session_state.get("trigger_processing", False):
        st.session_state.trigger_processing = False 
        query_to_process = st.session_state.get("query_to_process", "")

        if query_to_process:
            progress_area = st.container() 
            with progress_area:
                st.markdown("---")
                st.info("🚀 Processing your request...") 
                progress_bar = st.progress(0)
                simulated_steps = ["Initializing...", "Fetching data...", "Analyzing...", "Compiling brief..."]
                for i, step_name in enumerate(simulated_steps):
                    show_progress_step(step_name, random.randint(300, 600)) 
                    progress_bar.progress((i + 1) * (100 // len(simulated_steps)))
            
            orch_params = {
                "mode": st.session_state.mode,
                "session_id": st.session_state.session_id,
                "model": st.session_state.model_selections.get("llm", {}).get("model"),
                "temperature": st.session_state.model_selections.get("llm", {}).get("temperature"),
                "max_tokens": st.session_state.model_selections.get("llm", {}).get("max_tokens"),
                "analysis_provider": st.session_state.model_selections.get("analysis", {}).get("provider"),
                "include_correlations": st.session_state.model_selections.get("analysis", {}).get("include_correlations"),
                "include_risk_metrics": st.session_state.model_selections.get("analysis", {}).get("include_risk_metrics"),
                "tickers": st.session_state.tickers, 
                "news_limit": news_limit, # Value from slider via session_state key
                "retrieve_k": retrieve_k, # Value from slider via session_state key
                "topic": "finance, stocks" 
            }
            
            if st.session_state.mode == "voice":
                voice_settings = st.session_state.model_selections.get("voice", {})
                orch_params.update({
                    "stt_provider": voice_settings.get("stt_provider"),
                    "tts_provider": voice_settings.get("tts_provider"),
                    "voice": voice_settings.get("voice"),
                    "speaking_rate": voice_settings.get("speaking_rate"),
                    "pitch": voice_settings.get("pitch")
                })
                if st.session_state.audio_bytes:
                    import base64 
                    orch_params["audio_bytes_b64"] = base64.b64encode(st.session_state.audio_bytes).decode("utf-8")

            for ticker in st.session_state.tickers: 
                if ticker.lower() in query_to_process.lower():
                    orch_params["symbols_mentioned"] = ticker 
                    break
            if "brief" in query_to_process.lower() or "summary" in query_to_process.lower():
                orch_params["topic"] = "market summary, financial news"
                orch_params["news_limit"] = max(orch_params["news_limit"], 5)
            
            with st.spinner("🤖 FinAI is thinking... Generating your financial brief..."):
                try:
                    response = call_orchestrator(query_to_process, orch_params)
                    progress_bar.progress(100) 
                    progress_area.empty() 
                    
                    if response:
                        st.session_state.last_response = response
                        output_text = response.get("output", "Sorry, I couldn't generate a response.")
                        
                        for step in response.get("steps", []):
                            if step.get("tool") == "analysis_agent" and "response" in step:
                                st.session_state.last_analysis = step["response"]
                                break 
                        
                        st.header("📈 Your Financial Brief")
                        render_card("FinAI Response", output_text, icon="fa-lightbulb", card_type="info")
                        
                        audio_output_b64 = response.get("audio_output_b64") 
                        if st.session_state.mode == "voice" and audio_output_b64:
                            try:
                                import base64 
                                audio_bytes_out = base64.b64decode(audio_output_b64)
                                render_audio_player(audio_bytes_out, autoplay=True)
                            except Exception as e:
                                st.warning(f"Could not play audio response: {e}")
                        
                        result_tabs_titles = ["Agent Steps", "Raw Response", "Errors"]
                        res_tabs = st.tabs(result_tabs_titles)
                        
                        with res_tabs[0]: 
                            steps = response.get("steps", [])
                            if steps:
                                st.markdown("##### Process Details")
                                for i, step in enumerate(steps):
                                    tool_name = step.get("tool", f"Step {i+1}")
                                    latency = step.get("latency_ms", "N/A")
                                    st.markdown(f"**{tool_name}** (Latency: {latency}ms)")
                                    step_resp = step.get("response")
                                    if isinstance(step_resp, dict):
                                        summary_text = step_resp.get("summary", step_resp.get("text", str(step_resp)[:150] + "..."))
                                        st.caption(f"Output: {str(summary_text)[:200]}...")
                                    elif isinstance(step_resp, str):
                                        st.caption(f"Output: {step_resp[:200]}...")
                            else:
                                st.caption("No process steps recorded.")
                        
                        with res_tabs[1]: 
                            st.json(response, expanded=False)
                            
                        with res_tabs[2]: 
                            errors = response.get("errors", [])
                            if errors:
                                st.error("The following errors occurred during processing:")
                                for error in errors:
                                    st.code(f"Tool: {error.get('tool', 'Unknown')}\nError: {error.get('message', 'Unknown error')}", language="text")
                            else:
                                st.success("No errors reported during processing. ✅")
                        
                        st.session_state.history.append({
                            "query": query_to_process, "response": output_text,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "params": {k: v for k, v in orch_params.items() if k != "audio_bytes_b64"} 
                        })
                        
                        st.markdown("--- \n ##### Export Options")
                        export_cols = st.columns(3)
                        pdf_bytes = generate_pdf(output_text, "financial_brief.pdf")
                        export_cols[0].download_button("📄 Download PDF", pdf_bytes, f"FinAI_Brief_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf", use_container_width=True)
                        export_cols[1].download_button("📝 Download Text", output_text, f"FinAI_Brief_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
                        if st.session_state.mode == "voice" and audio_output_b64: 
                            try:
                                import base64
                                audio_dl_bytes = base64.b64decode(audio_output_b64)
                                export_cols[2].download_button("🔊 Download Audio", audio_dl_bytes, f"FinAI_Audio_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3", "audio/mp3", use_container_width=True)
                            except: pass #NOSONAR

                    else:
                        st.error("❌ Orchestrator did not return a valid response.")
                        
                except Exception as e:
                    st.error(f"⚠️ An application error occurred: {str(e)}")
                    st.exception(e) 
                
                st.session_state.user_input_text = ""
                st.session_state.transcript_text = ""
                st.session_state.audio_bytes = None 
                st.session_state.last_audio_bytes_processed = None

    query_to_process = st.session_state.get("query_to_process", "") # Re-fetch for safety, though it's cleared above
    if st.session_state.history and not st.session_state.get("trigger_processing", False) and not query_to_process :
        with st.expander("📜 Query History", expanded=False):
            def show_list_view():
                for i, item in enumerate(reversed(st.session_state.history)):
                    with st.container():
                        st.markdown(f"<small><b>{item['timestamp']}</b> - Query: <i>{item['query'][:60]}...</i></small>", unsafe_allow_html=True)
                        st.caption(f"Response: {item['response'][:120]}...")
                        if st.button("View Details", key=f"hist_detail_{i}", type="secondary"):
                            st.info(f"Full Query: {item['query']}")
                            render_card("Archived Response", item['response'], card_type="default")
                        st.markdown("---")
            
            def show_table_view():
                history_df_data = [{
                    "Time": item["timestamp"], "Query": item["query"],
                    "Mode": item["params"].get("mode", "N/A").capitalize()
                } for item in st.session_state.history]
                st.dataframe(pd.DataFrame(history_df_data).iloc[::-1], use_container_width=True, hide_index=True) 
            
            render_tabs({"List View": show_list_view, "Table View": show_table_view})

if __name__ == "__main__":
    main()