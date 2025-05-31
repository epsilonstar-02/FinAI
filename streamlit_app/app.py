# streamlit_app/app.py

import os
import uuid
import numpy as np # For AudioRecorderProcessor
import streamlit as st
import pandas as pd # For sample dashboard data
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import random
import base64 # For encoding audio to b64

# streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration # Added RTCConfiguration
import av # For audio frame processing

# Import from local modules
from .components import ( # Assuming components.py is in the same directory (streamlit_app)
    render_card, render_audio_player, show_progress_step, 
    render_stock_info, render_market_chart, render_tabs
)
from .advanced_components import render_model_selector, render_analysis_dashboard
from .utils import call_orchestrator, generate_pdf # call_stt removed, call_tts removed

# --- Audio Recording Settings & State ---
WEBRTC_SAMPLE_RATE = 16000 # Target sample rate for STT
WEBRTC_CHANNELS = 1

# Initialize session state for audio early if not present
if "audio_buffer_webrtc" not in st.session_state: # Renamed to avoid confusion
    st.session_state.audio_buffer_webrtc = []
if "last_recorded_audio_bytes" not in st.session_state: # Stores bytes from one recording
    st.session_state.last_recorded_audio_bytes = None
if "is_recording_webrtc" not in st.session_state: # Explicit state for WebRTC recorder
    st.session_state.is_recording_webrtc = False
if "webrtc_key_suffix" not in st.session_state: # To force remount of webrtc_streamer
    st.session_state.webrtc_key_suffix = 0


class AudioRecorderProcessor(AudioProcessorBase):
    """Collects audio frames from WebRTC and provides them as bytes."""
    def __init__(self) -> None:
        super().__init__()
        self._frames_buffer: List[np.ndarray] = []
        self.target_sample_rate = WEBRTC_SAMPLE_RATE
        self.target_channels = WEBRTC_CHANNELS

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Resample and convert to mono here if necessary using frame.reformat()
        # For simplicity, assuming browser sends something usable or STT agent handles variations.
        # A more robust approach resamples here to ensure consistency.
        # Example: desired_layout = av.AudioLayout("mono")
        #          desired_format = av.AudioFormat("s16") # Signed 16-bit
        #          resampled_frame = frame.reformat(format=desired_format, layout=desired_layout, rate=self.target_sample_rate)
        #          self._frames_buffer.append(resampled_frame.to_ndarray())
        
        # Current approach: collect as is, then process in get_recorded_data_and_clear
        self._frames_buffer.append(frame.to_ndarray())
        return frame # Must return the frame

    def get_recorded_data_and_clear(self) -> Optional[bytes]:
        if not self._frames_buffer:
            return None
        
        try:
            # Concatenate all frames: expected shape (num_channels, num_samples)
            raw_audio_data = np.concatenate(self._frames_buffer, axis=1)
            self._frames_buffer.clear()

            # Convert to pydub AudioSegment for easier manipulation
            # Determine format from raw_audio_data.dtype and shape
            # Assuming data is float32 from WebRTC common practice, scaled -1 to 1.
            # If not, AudioSegment.from_ndarray might need different params.
            
            # For pydub, we need to know original sample rate and channels from the frame.
            # This info is ideally obtained from the av.AudioFrame properties if they were stored.
            # For now, let's assume a common WebRTC output like 48kHz stereo float32.
            # This part is tricky without knowing exact input frame format from browser/webrtc.
            
            # Let's assume input frames are already mono, 16kHz, int16 after some browser processing
            # OR that pydub can handle it.
            # A more robust way is to use av.AudioFrame.sample_rate, .layout.name, .format.name
            # and pass that to AudioSegment.from_raw or by converting to standard format first.

            # Simplification: Assume input is float, needs conversion to int16 bytes.
            # This also assumes mono. If stereo, take one channel.
            if raw_audio_data.ndim > 1 and raw_audio_data.shape[0] > 1: # Multi-channel
                mono_audio_data = raw_audio_data[0, :] # Take first channel
            else:
                mono_audio_data = raw_audio_data.flatten()

            # Convert float audio (assumed -1 to 1 range) to int16
            if np.issubdtype(mono_audio_data.dtype, np.floating):
                int16_audio_data = (mono_audio_data * 32767).astype(np.int16)
            elif mono_audio_data.dtype == np.int16:
                int16_audio_data = mono_audio_data
            else: # Try to convert to int16 if some other integer type
                st.warning(f"Unexpected audio data type: {mono_audio_data.dtype}. Attempting conversion.")
                int16_audio_data = mono_audio_data.astype(np.int16)

            # At this point, int16_audio_data is a 1D numpy array of int16 samples.
            # The original sample rate of these frames is needed.
            # If not directly available, we have to make an assumption or try to detect.
            # Let's assume the browser provided it at a common rate like 48000 or the desired WEBRTC_SAMPLE_RATE.
            # For this example, we will convert to WAV bytes and then use pydub to ensure final format.
            
            temp_wav_io = io.BytesIO()
            # Assume input frame rate, e.g. 48000, will be resampled by pydub
            # This is a guess for frame.sample_rate if not explicitly handled in recv
            # If frames are directly at WEBRTC_SAMPLE_RATE, this is simpler.
            assumed_input_frame_rate = 48000 # Common browser output rate for WebRTC

            with wave.open(temp_wav_io, 'wb') as wf:
                wf.setnchannels(1) # Mono
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(assumed_input_frame_rate) # This is crucial
                wf.writeframes(int16_audio_data.tobytes())
            temp_wav_io.seek(0)
            
            # Use pydub to load this WAV and resample/reformat if needed
            audio_segment = AudioSegment.from_wav(temp_wav_io)
            
            # Ensure target format (16kHz, mono, 16-bit for STT typically)
            if audio_segment.frame_rate != self.target_sample_rate:
                audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
            if audio_segment.channels != self.target_channels:
                audio_segment = audio_segment.set_channels(self.target_channels)
            if audio_segment.sample_width != 2: # 16-bit
                audio_segment = audio_segment.set_sample_width(2)

            output_wav_io = io.BytesIO()
            audio_segment.export(output_wav_io, format="wav")
            return output_wav_io.getvalue()

        except Exception as e:
            st.error(f"Error processing recorded audio: {e}")
            logger.error(f"Audio processing error in WebRTC processor: {e}", exc_info=True)
            return None


# --- Load Environment Variables & Page Config ---
load_dotenv()
ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8004") # From utils.py context
VOICE_URL = os.getenv("VOICE_URL", "http://voice_agent:8006") # From utils.py context

if os.getenv("STREAMLIT_ENV", "prod").lower() == "dev":
    if ORCH_URL == "http://orchestrator:8004": ORCH_URL = "http://localhost:8004"
    if VOICE_URL == "http://voice_agent:8006": VOICE_URL = "http://localhost:8006"

st.set_page_config(
    page_title="FinAI - Financial Intelligence Assistant",
    page_icon="💹", layout="wide", initial_sidebar_state="expanded"
)

# Load CSS (ensure path is correct relative to app.py)
try:
    with open(os.path.join(os.path.dirname(__file__), "styles.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("`styles.css` not found. App styling may be affected.")


# --- Session State Initialization ---
def initialize_session_state():
    defaults = {
        "history": [], # Chat history
        "current_mode": "text", # "text" or "voice"
        "active_theme": "light", # "light" or "dark"
        "session_id": str(uuid.uuid4()),
        "selected_tickers": ["AAPL", "MSFT", "GOOGL"], # Default tickers for context
        "last_orchestrator_response": None, # Store full orchestrator response
        "last_analysis_data": None, # Specifically store analysis agent output
        "user_text_input": "", # For text_area persistence
        # "current_transcript": "", # No longer needed if STT is by orchestrator
        "model_configs": {}, # Stores output from render_model_selector
        "sidebar_news_limit": 3, # Default from original slider
        "sidebar_retrieve_k": 5, # Default from original slider
        "processing_triggered": False, # Flag to trigger orchestrator call
        "query_for_processing": "", # Stores the input query for processing
        "audio_for_processing_b64": None, # Stores base64 audio for orchestrator
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Theme Toggle (JS based, Python one is for direct calls if needed) ---
def apply_theme_js(): # Renamed for clarity
    # This JS directly manipulates body attribute, CSS :root vars then apply.
    # Streamlit's own theme changes might also re-render, ensure compatibility.
    theme_js = f"""
        <script>
            const body = document.querySelector('body');
            function applyTheme(theme) {{
                body.setAttribute('data-theme', theme);
                console.log('Applied theme:', theme);
            }}
            // Apply initial theme from Streamlit session_state
            applyTheme('{st.session_state.get("active_theme", "light")}');

            // Function for button to call
            window.toggleStreamlitTheme = function() {{
                const currentTheme = body.getAttribute('data-theme');
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                applyTheme(newTheme);
                // Inform Streamlit about the change if Python state needs update
                // This is hard without Python<->JS communication framework beyond basic JS.
                // For now, JS handles visual, Python state is separate.
                // If python_toggle_theme is called, it will update session_state and rerun.
            }}
        </script>
    """
    st.markdown(theme_js, unsafe_allow_html=True)

def python_toggle_theme_action(): # Action for a Streamlit button if used
    st.session_state.active_theme = "dark" if st.session_state.active_theme == "light" else "light"
    # No st.rerun() here, let Streamlit handle it if button causes rerun.
    # apply_theme_js() will be called on next render and pick up new active_theme.

# --- UI Rendering Functions ---
def display_dashboard_tab():
    st.subheader("Market Overview (Sample Data)")
    cols = st.columns(len(st.session_state.selected_tickers[:5])) # Use selected_tickers
    for i, ticker in enumerate(st.session_state.selected_tickers[:5]):
        with cols[i]:
            # Replace with actual API calls if desired, or keep random for demo
            price = random.uniform(50, 600) 
            change = random.uniform(-0.05, 0.05) # As decimal percentage
            render_stock_info(ticker, price, change) # change is passed as decimal
    
    st.subheader("Market Trends (Sample Chart Data)")
    dates = pd.to_datetime([(datetime.now() - timedelta(days=i)) for i in range(89, -1, -1)]) # 90 days
    chart_data = pd.DataFrame({
        'date': dates,
        'price': [round(150 + i * 0.5 + random.uniform(-15, 15), 2) for i in range(90)]
    })
    render_market_chart(chart_data) # Uses defaults for column names


def display_portfolio_analysis_tab():
    if st.session_state.last_analysis_data:
        render_analysis_dashboard(st.session_state.last_analysis_data)
    else:
        st.info("No portfolio analysis data available. Use the 'Financial Assistant' tab to generate insights or generate a sample below.")
        if st.button("🔬 Generate Sample Portfolio Analysis", key="gen_sample_analysis_btn"):
            with st.spinner("Generating sample analysis... This may take a moment."):
                # Construct a sample payload for the orchestrator's analysis capabilities
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
                prices = {sym: random.uniform(100, 2000) for sym in symbols}
                historical_data = {}
                for sym in symbols:
                    base_p = prices[sym]
                    hist_dps = []
                    for i in range(60, 0, -1): # 60 days of history
                        day = (datetime.now() - timedelta(days=i))
                        close_p = round(max(1, base_p * (1 - 0.0008 * i + random.uniform(-0.02, 0.02))), 2)
                        hist_dps.append({"date": day.strftime("%Y-%m-%d"), "close": close_p, "open": close_p*0.99, "high": close_p*1.01, "low": close_p*0.98, "volume": random.randint(100000,5000000)})
                    historical_data[sym] = hist_dps
                
                # Parameters for the orchestrator, specifically for analysis
                # These will go into RunRequest.params
                analysis_call_params = {
                    "symbols": symbols, # Hint for API agent if it needs to fetch symbols too
                    "prices": prices,   # Actual prices for analysis agent
                    "historical": historical_data, # Actual historical for analysis
                    "analysis_provider": st.session_state.model_configs.get("analysis_config", {}).get("provider", "advanced"),
                    "include_correlations": st.session_state.model_configs.get("analysis_config", {}).get("include_correlations", True),
                    "include_risk_metrics": st.session_state.model_configs.get("analysis_config", {}).get("include_risk_metrics", True),
                }
                
                # The input query for the orchestrator.
                # For sample analysis, the query might just instruct to analyze the provided data.
                sample_query = f"Provide a detailed financial analysis for the portfolio: {', '.join(symbols)}."
                
                response = call_orchestrator(sample_query, "text", analysis_call_params) # Mode text, params contain data
                
                if response and not response.get("error"):
                    st.session_state.last_orchestrator_response = response
                    analysis_step_data = None
                    for step in response.get("steps", []):
                        if step.get("tool") == "analysis_agent" and isinstance(step.get("response"), dict):
                            analysis_step_data = step["response"]
                            # Ensure success flag from agent if present
                            if analysis_step_data.get("success", True) is False : # explicit check for False
                                st.warning(f"Analysis Agent reported an issue: {analysis_step_data.get('detail','Unknown error')}")
                                analysis_step_data = None # Do not use failed analysis data
                            break
                    
                    if analysis_step_data:
                        st.session_state.last_analysis_data = analysis_step_data
                        st.success("Sample analysis generated and displayed!")
                        st.rerun() # To re-render the tab with new data
                    else:
                        st.warning("Analysis data not found in orchestrator response or analysis agent failed.")
                        if response.get("errors"): st.error(f"Orchestrator errors: {response.get('errors')}")
                else:
                    st.error(f"Failed to generate sample analysis. Orchestrator error: {response.get('message', 'Unknown error') if response else 'No response'}")


def display_financial_assistant_tab():
    st.header("Financial Intelligence Assistant")
    
    # Mode selection buttons
    mode_cols = st.columns(2)
    button_type_text = "primary" if st.session_state.current_mode == "text" else "secondary"
    button_type_voice = "primary" if st.session_state.current_mode == "voice" else "secondary"

    if mode_cols[0].button("💬 Text Mode", use_container_width=True, type=button_type_text, key="text_mode_btn"):
        if st.session_state.current_mode != "text": 
            st.session_state.current_mode = "text"; st.rerun()
    if mode_cols[1].button("🎙️ Voice Mode", use_container_width=True, type=button_type_voice, key="voice_mode_btn"):
        if st.session_state.current_mode != "voice":
            st.session_state.current_mode = "voice"; st.rerun()
            st.session_state.webrtc_key_suffix +=1 # Force remount of webrtc

    st.caption(f"Current Mode: **{st.session_state.current_mode.upper()}**")
    st.markdown("---")

    query_to_send_to_orchestrator = ""
    audio_b64_for_orchestrator = None

    if st.session_state.current_mode == "text":
        st.session_state.user_text_input = st.text_area(
            "Enter your financial query:", value=st.session_state.user_text_input, height=100,
            placeholder="E.g., What's the current market outlook for tech stocks? Or, price of AAPL?"
        )
        query_to_send_to_orchestrator = st.session_state.user_text_input

        # Query suggestions
        st.markdown("<p style='font-size: 0.9em; margin-bottom: 0.5em;'><strong>💡 Try asking:</strong></p>", unsafe_allow_html=True)
        suggestions = ["Price of MSFT and NVDA?", "Latest news on semiconductor industry.", "Analyze portfolio: AAPL, GOOGL."]
        sugg_cols = st.columns(len(suggestions))
        for i, sugg_text in enumerate(suggestions):
            if sugg_cols[i].button(sugg_text, key=f"sugg_{i}", use_container_width=True):
                st.session_state.user_text_input = sugg_text
                st.rerun() # To update the text_area and prepare for submission
    
    else: # Voice Mode
        st.markdown("##### Record Your Query")
        st.caption("Click 'Start Recording', speak your query, then click 'Stop Recording'.")

        if "webrtc_processor_instance" not in st.session_state:
             st.session_state.webrtc_processor_instance = None
        
        def webrtc_audio_processor_factory():
            # This factory is called by webrtc_streamer, store instance for later access
            instance = AudioRecorderProcessor()
            st.session_state.webrtc_processor_instance = instance
            return instance

        # WebRTC buttons and component
        rec_col1, rec_col2 = st.columns(2)
        if rec_col1.button("🎙️ Start Recording", key="webrtc_start", use_container_width=True, disabled=st.session_state.is_recording_webrtc):
            st.session_state.is_recording_webrtc = True
            st.session_state.last_recorded_audio_bytes = None # Clear previous
            st.session_state.webrtc_key_suffix +=1 # Force remount
            st.rerun()
        
        if rec_col2.button("⏹️ Stop Recording", key="webrtc_stop", use_container_width=True, disabled=not st.session_state.is_recording_webrtc):
            st.session_state.is_recording_webrtc = False
            # Processor instance might not be available immediately after stopping,
            # it's better to get data when is_recording_webrtc becomes false AND processor exists
            st.rerun()

        if st.session_state.is_recording_webrtc:
            # RTCConfiguration for STUN servers (helps with NAT traversal)
            RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_ctx = webrtc_streamer(
                key=f"audio_recorder_{st.session_state.webrtc_key_suffix}", # Changing key forces remount
                mode=WebRtcMode.SENDONLY, # Send only for recording
                audio_processor_factory=webrtc_audio_processor_factory,
                media_stream_constraints={"video": False, "audio": True},
                rtc_configuration=RTC_CONFIGURATION,
                # desired_playing_state=st.session_state.is_recording_webrtc # Handled by remount via key change
            )
            if not webrtc_ctx.state.playing: # If it stopped for some reason (e.g. permissions denied)
                st.session_state.is_recording_webrtc = False # Reset state
                # st.rerun() # Rerun if state was unexpectedly changed

        # Process recorded audio when recording stops
        if not st.session_state.is_recording_webrtc and st.session_state.webrtc_processor_instance:
            recorded_bytes = st.session_state.webrtc_processor_instance.get_recorded_data_and_clear()
            st.session_state.webrtc_processor_instance = None # Clear instance after getting data
            if recorded_bytes:
                st.session_state.last_recorded_audio_bytes = recorded_bytes
                st.success("✅ Audio recorded successfully!")
                st.audio(st.session_state.last_recorded_audio_bytes, format="audio/wav", sample_rate=WEBRTC_SAMPLE_RATE)
                # Audio is now ready for orchestrator
            # Do not rerun here, let user click "Generate"
        
        if st.session_state.last_recorded_audio_bytes:
            query_to_send_to_orchestrator = "Process the recorded audio query." # Placeholder for orchestrator
            audio_b64_for_orchestrator = base64.b64encode(st.session_state.last_recorded_audio_bytes).decode('utf-8')
            st.info("Audio ready to be sent for processing.")
        elif st.session_state.is_recording_webrtc: # If still recording
             pass # UI already shows recorder
        else: # Not recording, no audio yet
             st.info("Click 'Start Recording' to record your voice query.")


    # Submit button for both modes
    # Disable if text mode has no input, or voice mode has no recorded audio
    can_submit = False
    if st.session_state.current_mode == "text" and query_to_send_to_orchestrator.strip():
        can_submit = True
    elif st.session_state.current_mode == "voice" and audio_b64_for_orchestrator:
        can_submit = True

    if st.button("🚀 Generate Financial Brief", type="primary", use_container_width=True, disabled=not can_submit, key="generate_brief_btn"):
        st.session_state.processing_triggered = True
        st.session_state.query_for_processing = query_to_send_to_orchestrator
        st.session_state.audio_for_processing_b64 = audio_b64_for_orchestrator # Will be None in text mode
        st.rerun()


# --- Main Application Logic & UI ---
def main():
    initialize_session_state()
    apply_theme_js() # Apply theme based on session_state on each run

    # App Header (Improved with JS theme toggle directly in header)
    st.markdown(f"""
    <div class="app-header">
        <img src="app/static/logo.png" class="app-logo" alt="FinAI Logo" /> 
        <div>
            <h1 class="app-title">FinAI</h1>
            <p class="subtitle">Advanced Multi-Agent Financial Intelligence</p>
        </div>
        <div style="margin-left: auto; display: flex; align-items: center;">
            <button onclick="window.toggleStreamlitTheme()" title="Toggle Theme" class="theme-toggle-button">
                <i class="fas fa-adjust"></i>
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load static logo (ensure path is correct for Streamlit deployment)
    # Streamlit typically serves from a 'static' folder if app.py is in root of that structure.
    # Or use st.image with a local path. For now, assuming the HTML img src works.
    # If running `streamlit run streamlit_app/app.py`, `app/static/logo.png` might need to be `static/logo.png`
    # or adjust path in markdown if logo is in `streamlit_app/assets/`. Let's assume `streamlit_app/assets/`.
    # For `st.image`, path relative to `app.py`:
    # logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    # if os.path.exists(logo_path): st.sidebar.image(logo_path, width=70) # Example in sidebar
    # The HTML approach for header needs logo to be accessible via a URL path Streamlit serves.


    # --- Sidebar Configuration ---
    with st.sidebar:
        logo_path_sidebar = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path_sidebar): st.image(logo_path_sidebar, width=60)
        else: st.markdown("## FinAI") # Fallback text logo
        
        st.header("Settings & Controls")
        st.markdown("---")

        with st.expander("🤖 Model & Analysis Settings", expanded=False):
            # render_model_selector now returns a structured dict
            st.session_state.model_configs = render_model_selector(show_subheader=False)
        
        st.markdown("---")
        st.subheader("Context Parameters")
        # Tickers for context
        all_ticker_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "BTC-USD", "ETH-USD", "EURUSD=X"]
        st.session_state.selected_tickers = st.multiselect(
            "Ticker Symbols for Contextual Data", options=all_ticker_options,
            default=st.session_state.get("selected_tickers", ["AAPL", "MSFT", "GOOGL"]), # Use .get for safety
            key="tickers_multiselect"
        )
        # Other context params (sliders store directly into session_state via key)
        st.session_state.sidebar_news_limit = st.slider(
            "News Articles Limit (Scraping)", 1, 10, 
            value=st.session_state.get("sidebar_news_limit", 3), key="news_limit_slider_main"
        )
        st.session_state.sidebar_retrieve_k = st.slider(
            "Retrieved Documents Top-K (Retrieval)", 1, 10, 
            value=st.session_state.get("sidebar_retrieve_k", 5), key="retrieve_k_slider_main"
        )
        st.markdown("---")
        
        st.subheader("Session")
        st.caption(f"ID: {st.session_state.session_id[:8]}...")
        st.caption(f"History Count: {len(st.session_state.history)}")
        if st.button("Clear Chat History", key="clear_history_main_btn", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_orchestrator_response = None
            st.session_state.last_analysis_data = None
            st.session_state.user_text_input = ""
            st.session_state.last_recorded_audio_bytes = None
            st.session_state.audio_for_processing_b64 = None
            st.success("Chat history cleared.")
            time.sleep(0.5); st.rerun() # Brief pause then rerun
        
        st.markdown("---")
        st.info("FinAI uses multiple specialized agents to provide comprehensive financial intelligence.")
        st.caption(f"Version {st.get_option('client. streamlitAppVersion') or '1.0'} | Theme: {st.session_state.active_theme.capitalize()}")


    # --- Main Area Tabs ---
    tab_titles = ["📊 Market Overview", "📈 Portfolio Analysis", "🤖 Financial Assistant"]
    main_tabs = st.tabs(tab_titles)
    
    with main_tabs[0]: display_dashboard_tab()
    with main_tabs[1]: display_portfolio_analysis_tab()
    with main_tabs[2]: display_financial_assistant_tab()


    # --- Orchestrator Call and Response Handling (triggered by flag) ---
    if st.session_state.get("processing_triggered", False):
        st.session_state.processing_triggered = False # Reset trigger
        
        query_input = st.session_state.query_for_processing
        current_mode = st.session_state.current_mode
        audio_b64_input = st.session_state.audio_for_processing_b64 # None if text mode

        if (current_mode == "text" and not query_input.strip()) or \
           (current_mode == "voice" and not audio_b64_input):
            st.warning("Please provide input (text or voice recording) before generating.")
            return # Exit processing if no valid input for the mode

        progress_area = st.container()
        with progress_area:
            st.markdown("---")
            st.info("🚀 Processing your request with FinAI agents...")
            # Simulate steps (can be replaced with actual feedback if orchestrator provides it)
            sim_steps = ["Initializing Orchestration...", "Querying Data Agents...", "Performing Analysis...", "Generating Insights with LLM..."]
            prog_bar = st.progress(0)
            for i, step_desc in enumerate(sim_steps):
                show_progress_step(step_desc, is_complete=False, details="Working...") # Show as in-progress
                time.sleep(random.uniform(0.3, 0.7)) # Simulate work
                prog_bar.progress(int(((i + 1) / len(sim_steps)) * 100))
            show_progress_step("Finalizing brief...", is_complete=False, details="Almost done!")


        # Construct orchestrator_params (the `params` field of RunRequest)
        # This dict will contain all detailed configurations and contextual data.
        orchestrator_run_params: Dict[str, Any] = {
            "session_id": st.session_state.session_id, # Pass session ID
            # LLM settings from model_configs
            **st.session_state.model_configs.get("llm_config", {}),
            # Voice settings (STT for input by Orchestrator, TTS for output by Orchestrator)
            **st.session_state.model_configs.get("voice_config", {}),
            # Analysis settings
            **st.session_state.model_configs.get("analysis_config", {}),
            # Contextual data from sidebar
            "tickers": st.session_state.selected_tickers,
            "news_limit": st.session_state.sidebar_news_limit,
            "retrieve_k": st.session_state.sidebar_retrieve_k,
            # If main query contains symbols, Orchestrator might extract them.
            # If a specific topic is derived for news/retrieval, it can be added here.
            # "topic": "derived topic if any"
        }
        # Add audio input if in voice mode
        if current_mode == "voice" and audio_b64_input:
            orchestrator_run_params["audio_bytes_b64"] = audio_b64_input
            # The 'input_text' to orchestrator can be a generic instruction if audio is the main input
            if not query_input.strip(): query_input = "Transcribe and process the provided audio."
        
        # Call orchestrator
        with st.spinner("🤖 FinAI is coordinating agents to generate your brief... This may take some time."):
            orchestrator_response = call_orchestrator(query_input, current_mode, orchestrator_run_params)
        
        progress_area.empty() # Clear progress display

        if orchestrator_response and not orchestrator_response.get("error"):
            st.session_state.last_orchestrator_response = orchestrator_response
            output_text = orchestrator_response.get("output", "Sorry, no textual output was generated.")
            
            # Update last_analysis_data if analysis was part of this run
            for step in orchestrator_response.get("steps", []):
                if step.get("tool") == "analysis_agent" and isinstance(step.get("response"), dict):
                    st.session_state.last_analysis_data = step["response"]
                    break # Assume only one analysis step relevant for dashboard

            st.markdown("---")
            st.header("📈 Your Financial Briefing")
            render_card("FinAI Response", output_text, icon="fa-lightbulb", card_type="info")
            
            # Handle audio output from orchestrator (if any, and if in voice mode)
            audio_b64_output = orchestrator_response.get("audio_output_b64")
            if current_mode == "voice" and audio_b64_output:
                try:
                    audio_bytes_out = base64.b64decode(audio_b64_output)
                    render_audio_player(audio_bytes_out, autoplay=True) # Autoplay the response
                except Exception as e_audio_play:
                    st.warning(f"Could not play audio response: {e_audio_play}")
            
            # Display steps, raw response, errors using tabs
            res_tab_titles = ["Agent Process", "Full Response JSON", "Reported Errors"]
            res_tabs_rendered = st.tabs(res_tab_titles)
            
            with res_tabs_rendered[0]: # Agent Process
                steps_data = orchestrator_response.get("steps", [])
                if steps_data:
                    st.markdown("##### Orchestration Steps")
                    for i, step in enumerate(steps_data):
                        tool = step.get("tool", f"Unknown Step {i+1}")
                        latency = step.get("latency_ms", "N/A")
                        resp_summary = str(step.get("response", ""))[:200] + "..." if step.get("response") else "No response data."
                        if isinstance(step.get("response"), dict) and step["response"].get("success") is False:
                             st.error(f"**{tool}** (Latency: {latency}ms) - Failed: {step['response'].get('detail', resp_summary)}")
                        else:
                             st.success(f"**{tool}** (Latency: {latency}ms)")
                             with st.expander("Show Details", expanded=False): st.json(step.get("response"), expanded=False)
                else: st.caption("No processing steps recorded by orchestrator.")

            with res_tabs_rendered[1]: # Full Response JSON
                st.json(orchestrator_response, expanded=False)
                
            with res_tabs_rendered[2]: # Reported Errors
                errors_data = orchestrator_response.get("errors", [])
                if errors_data:
                    st.error("The following issues were reported during processing:")
                    for err_item in errors_data:
                        st.code(f"Tool: {err_item.get('tool', 'N/A')}\nMessage: {err_item.get('message', 'Unknown issue')}", language="text")
                else: st.success("✅ No errors reported by the orchestrator.")

            # Add to history
            st.session_state.history.append({
                "query": query_input, # This is text query or "Process audio..."
                "mode": current_mode,
                "response_text": output_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "params_sent": {k:v for k,v in orchestrator_run_params.items() if k != "audio_bytes_b64"} # Don't store large audio in history
            })
            
            # Export options
            st.markdown("--- \n ##### Export Options")
            export_cols = st.columns(3 if (current_mode == "voice" and audio_b64_output) else 2)
            try:
                pdf_b = generate_pdf(output_text)
                export_cols[0].download_button(
                    "📄 Download PDF", pdf_b, f"FinAI_Brief_{datetime.now():%Y%m%d_%H%M}.pdf", 
                    "application/pdf", use_container_width=True
                )
            except Exception as e_pdf: st.error(f"PDF generation failed: {e_pdf}")

            export_cols[1].download_button(
                "📝 Download Text", output_text, f"FinAI_Brief_{datetime.now():%Y%m%d_%H%M}.txt", 
                "text/plain", use_container_width=True
            )
            if current_mode == "voice" and audio_b64_output:
                try:
                    audio_dl_bytes = base64.b64decode(audio_b64_output)
                    export_cols[2].download_button(
                        "🔊 Download Audio Response", audio_dl_bytes, f"FinAI_Audio_{datetime.now():%Y%m%d_%H%M}.mp3", 
                        "audio/mp3", use_container_width=True)
                except Exception as e_audio_dl: st.warning(f"Audio download prep failed: {e_audio_dl}")
        
        else: # Orchestrator call failed (error handled by call_orchestrator and displayed)
            st.error(f"Orchestrator call failed. Message: {orchestrator_response.get('message', 'No specific message.') if orchestrator_response else 'No response from orchestrator.'}")
            if orchestrator_response and orchestrator_response.get("details"):
                st.expander("Error Details").code(orchestrator_response.get("details"))

        # Clear inputs for next round
        st.session_state.user_text_input = ""
        st.session_state.last_recorded_audio_bytes = None
        st.session_state.audio_for_processing_b64 = None


    # --- Display Query History (if not processing and history exists) ---
    if not st.session_state.get("processing_triggered", False) and st.session_state.history:
        with st.expander("📜 Query History", expanded=False):
            history_views = {
                "List View": lambda: [
                    render_card(
                        f"Query ({item['mode']}) @ {item['timestamp']}", 
                        f"**Q:** {item['query'][:100]}...\n**A:** {item['response_text'][:150]}...",
                        icon="fa-history", card_type="default", timestamp_override=item['timestamp']
                    ) for item in reversed(st.session_state.history) # Show newest first
                ],
                "Table View": lambda: st.dataframe(
                    pd.DataFrame(st.session_state.history).iloc[::-1], # Reverse for newest first
                    use_container_width=True, hide_index=True,
                    column_config={ # Example column config
                        "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                        "query": st.column_config.TextColumn("Query", width="medium"),
                        "response_text": st.column_config.TextColumn("Response Preview", width="large", help="First 150 chars"),
                        "mode": st.column_config.TextColumn("Mode")
                    }
                )
            }
            render_tabs(history_views)


if __name__ == "__main__":
    main()