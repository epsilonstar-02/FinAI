"""Advanced components for the FinAI Streamlit application."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import altair as alt
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns


def render_provider_selector(section_title, providers, default=None, key_prefix=""):
    """
    Render a provider selection interface with radio buttons.
    
    Args:
        section_title: Title for the provider section
        providers: List of available providers
        default: Default selected provider
        key_prefix: Prefix for the session state key
    
    Returns:
        Selected provider
    """
    st.subheader(section_title)
    
    # Create columns for each provider
    cols = st.columns(len(providers))
    
    # Session state key
    state_key = f"{key_prefix}_provider"
    if state_key not in st.session_state:
        st.session_state[state_key] = default or providers[0]
    
    # Create radio buttons
    for i, provider in enumerate(providers):
        with cols[i]:
            provider_selected = st.radio(
                f"{provider}",
                [provider],
                key=f"{key_prefix}_{provider}",
                index=0 if st.session_state[state_key] == provider else None,
                label_visibility="collapsed"
            )
            
            if provider_selected:
                st.session_state[state_key] = provider
                
            # Indicate selected with styling
            if st.session_state[state_key] == provider:
                st.markdown(f"<div class='selected-provider'>{provider}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='provider-option'>{provider}</div>", unsafe_allow_html=True)
    
    return st.session_state[state_key]


def render_correlation_matrix(correlation_data):
    """
    Render a correlation matrix heatmap.
    
    Args:
        correlation_data: Dictionary of dictionaries with correlation values
        
    Returns:
        Plotly figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(correlation_data)
    
    # Create heatmap
    fig = px.imshow(
        df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Asset Correlation Matrix"
    )
    
    fig.update_layout(
        width=600,
        height=500,
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", 
            thickness=20,
            lenmode="pixels", 
            len=400,
        )
    )
    
    return fig


def render_risk_metrics_radar(risk_metrics, ticker):
    """
    Render a radar chart of risk metrics for a specific ticker.
    
    Args:
        risk_metrics: Dictionary of risk metrics for the ticker
        ticker: Ticker symbol
    
    Returns:
        Plotly figure object
    """
    if ticker not in risk_metrics:
        return None
    
    metrics = risk_metrics[ticker]
    
    # Normalize metrics for radar chart (0-1 scale)
    metrics_to_plot = {
        "Sharpe Ratio": min(max(metrics.get("sharpe_ratio", 0), 0) / 3, 1),  # Good Sharpe is 1-3
        "Sortino Ratio": min(max(metrics.get("sortino_ratio", 0), 0) / 3, 1),  # Good Sortino is 1-3
        "1-Max Drawdown": 1 - min(metrics.get("max_drawdown", 0), 1),  # Lower drawdown is better
        "1-VaR": 1 - min(abs(metrics.get("var_95", 0)), 0.2) / 0.2,  # Lower VaR is better
        "Beta (1.0 is neutral)": 1 - min(abs(metrics.get("beta", 1) - 1), 1)  # Closer to 1 is more neutral
    }
    
    categories = list(metrics_to_plot.keys())
    values = list(metrics_to_plot.values())
    
    # Close the loop for the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=ticker,
        line_color='rgba(45, 146, 195, 0.8)',
        fillcolor='rgba(45, 146, 195, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=f"Risk Profile: {ticker}"
    )
    
    return fig


def render_portfolio_exposures(exposures):
    """
    Render a pie chart of portfolio exposures.
    
    Args:
        exposures: Dictionary of asset symbols to exposure percentages
        
    Returns:
        Plotly figure object
    """
    labels = list(exposures.keys())
    values = [exposures[symbol] * 100 for symbol in labels]  # Convert to percentages
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.Plotly)
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    return fig


def render_price_changes(changes):
    """
    Render a horizontal bar chart of price changes.
    
    Args:
        changes: Dictionary of asset symbols to percentage changes
        
    Returns:
        Plotly figure object
    """
    symbols = list(changes.keys())
    pct_changes = [changes[symbol] * 100 for symbol in symbols]  # Convert to percentages
    
    # Sort by change value
    sorted_indices = np.argsort(pct_changes)
    symbols = [symbols[i] for i in sorted_indices]
    pct_changes = [pct_changes[i] for i in sorted_indices]
    
    # Create colors based on positive/negative
    colors = ['green' if x >= 0 else 'red' for x in pct_changes]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=symbols,
        x=pct_changes,
        orientation='h',
        marker_color=colors,
        text=[f"{x:.2f}%" for x in pct_changes],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Daily Price Changes",
        xaxis_title="Percent Change",
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    return fig


def render_volatility_comparison(volatility):
    """
    Render a bar chart comparing volatility across assets.
    
    Args:
        volatility: Dictionary of asset symbols to volatility values
        
    Returns:
        Plotly figure object
    """
    symbols = list(volatility.keys())
    vol_values = [volatility[symbol] * 100 for symbol in symbols]  # Convert to percentages
    
    # Sort by volatility
    sorted_indices = np.argsort(vol_values)[::-1]  # Descending
    symbols = [symbols[i] for i in sorted_indices]
    vol_values = [vol_values[i] for i in sorted_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols,
        y=vol_values,
        marker_color='rgba(76, 114, 176, 0.8)',
        text=[f"{x:.2f}%" for x in vol_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Asset Volatility Comparison",
        yaxis_title="Volatility (%)",
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    return fig


def render_audio_recorder():
    """
    Render an audio recorder with visualization.
    
    Returns:
        Audio bytes if recorded
    """
    st.markdown("<div class='audio-recorder-container'>", unsafe_allow_html=True)
    st.markdown("<h3>Voice Input</h3>", unsafe_allow_html=True)
    
    # Create container for visualization
    visualizer_container = st.empty()
    visualizer_container.markdown("<div class='audio-visualizer'></div>", unsafe_allow_html=True)
    
    # Create columns for record and stop buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        record_button = st.button("🎙️ Record", key="record_button")
    
    with col2:
        stop_button = st.button("⏹️ Stop", key="stop_button")
    
    with col3:
        clear_button = st.button("🗑️ Clear", key="clear_button")
    
    # Inject JavaScript for audio recording
    st.markdown("""
    <script>
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        let mediaRecorder;
        let audioChunks = [];
        let recording = false;
        
        async function startRecording() {
            if (recording) return;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.addEventListener("dataavailable", event => {
                    audioChunks.push(event.data);
                });
                
                mediaRecorder.start();
                recording = true;
                
                // Set up visualization
                const source = audioContext.createMediaStreamSource(stream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                source.connect(analyser);
                
                // Visualize the audio
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                
                function draw() {
                    if (!recording) return;
                    
                    requestAnimationFrame(draw);
                    analyser.getByteFrequencyData(dataArray);
                    
                    const visualizer = document.querySelector('.audio-visualizer');
                    if (visualizer) {
                        visualizer.innerHTML = '';
                        
                        for (let i = 0; i < bufferLength; i++) {
                            const bar = document.createElement('div');
                            const height = dataArray[i] / 2;
                            bar.style.height = height + 'px';
                            bar.className = 'visualizer-bar';
                            visualizer.appendChild(bar);
                        }
                    }
                }
                
                draw();
            } catch (err) {
                console.error("Error accessing microphone:", err);
                alert("Could not access microphone. Please ensure you have granted permission.");
            }
        }
        
        async function stopRecording() {
            if (!recording || !mediaRecorder) return;
            
            mediaRecorder.stop();
            recording = false;
            
            const audioBlob = new Promise(resolve => {
                mediaRecorder.addEventListener("stop", () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    resolve(audioBlob);
                });
            });
            
            const blob = await audioBlob;
            const reader = new FileReader();
            
            reader.readAsDataURL(blob);
            reader.onloadend = function() {
                const base64data = reader.result.split(',')[1];
                window.parent.postMessage({
                    type: "audioRecorded",
                    data: base64data
                }, "*");
            };
            
            // Clear visualization
            const visualizer = document.querySelector('.audio-visualizer');
            if (visualizer) {
                visualizer.innerHTML = '';
            }
        }
        
        function clearRecording() {
            audioChunks = [];
            recording = false;
            
            // Clear visualization
            const visualizer = document.querySelector('.audio-visualizer');
            if (visualizer) {
                visualizer.innerHTML = '';
            }
            
            window.parent.postMessage({
                type: "audioClear"
            }, "*");
        }
        
        // Listen for button clicks from Streamlit
        window.addEventListener("message", (event) => {
            if (event.data.type === 'streamlit:componentReady') {
                // Setup listeners for our buttons
                const buttons = parent.document.querySelectorAll('button');
                
                buttons.forEach(button => {
                    button.addEventListener('click', function(e) {
                        if (this.innerText.includes('Record')) {
                            startRecording();
                        } else if (this.innerText.includes('Stop')) {
                            stopRecording();
                        } else if (this.innerText.includes('Clear')) {
                            clearRecording();
                        }
                    });
                });
            }
        });
    </script>
    
    <style>
        .audio-visualizer {
            display: flex;
            align-items: flex-end;
            height: 100px;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.1);
            border-radius: 4px;
            overflow: hidden;
            padding: 2px;
        }
        
        .visualizer-bar {
            width: 4px;
            background-color: #ff4b4b;
            margin: 0 1px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a placeholder for the recorded audio
    audio_placeholder = st.empty()
    
    # Create session state for recorded audio
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None
    
    # JavaScript callback handler
    st.markdown("""
    <script>
        window.addEventListener("message", (event) => {
            if (event.data.type === "audioRecorded") {
                const data = event.data.data;
                
                // Store the base64 audio data in Streamlit's session state
                // We need to use Streamlit's setComponentValue API
                if (window.parent.Streamlit) {
                    window.parent.Streamlit.setComponentValue({
                        type: "audio",
                        data: data
                    });
                }
            } else if (event.data.type === "audioClear") {
                if (window.parent.Streamlit) {
                    window.parent.Streamlit.setComponentValue({
                        type: "audioClear"
                    });
                }
            }
        });
    </script>
    """, unsafe_allow_html=True)
    
    return st.session_state.audio_bytes


def render_model_selector():
    """
    Render a model selector with tabs for different agent types.
    
    Returns:
        Dictionary with selected models for each agent type
    """
    st.subheader("Model Selection")
    
    model_tabs = st.tabs(["Language Models", "Voice Models", "Analysis Providers"])
    
    with model_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### LLM Provider")
            llm_provider = st.selectbox(
                "Select LLM Provider",
                ["Gemini", "OpenAI", "Anthropic", "HuggingFace"],
                key="llm_provider"
            )
        
        with col2:
            st.markdown("#### Model")
            # Show different models based on provider
            if llm_provider == "Gemini":
                llm_model = st.selectbox(
                    "Select Model",
                    ["gemini-flash", "gemini-pro", "gemini-1.5-pro"],
                    key="gemini_model"
                )
            elif llm_provider == "OpenAI":
                llm_model = st.selectbox(
                    "Select Model",
                    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                    key="openai_model"
                )
            elif llm_provider == "Anthropic":
                llm_model = st.selectbox(
                    "Select Model",
                    ["claude-2", "claude-instant", "claude-3-opus"],
                    key="anthropic_model"
                )
            else:
                llm_model = st.selectbox(
                    "Select Model",
                    ["llama-2-7b", "mistral-7b", "falcon-7b"],
                    key="hf_model"
                )
        
        # LLM parameters
        st.markdown("#### Parameters")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature")
        with col2:
            max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100, key="max_tokens")
    
    with model_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### STT Provider")
            stt_provider = st.selectbox(
                "Select STT Provider",
                ["Whisper", "Google Cloud", "Azure", "Vosk"],
                key="stt_provider"
            )
        
        with col2:
            st.markdown("#### TTS Provider")
            tts_provider = st.selectbox(
                "Select TTS Provider",
                ["Google TTS", "Edge TTS", "ElevenLabs", "Amazon Polly"],
                key="tts_provider"
            )
        
        # Voice parameters
        st.markdown("#### Voice Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            voice = st.selectbox(
                "Voice",
                ["en-US-Neural2-F", "en-US-Neural2-M", "en-GB-Neural2-F", "en-GB-Neural2-M"],
                key="voice"
            )
        with col2:
            speaking_rate = st.slider("Speaking Rate", 0.5, 2.0, 1.0, 0.1, key="speaking_rate")
        with col3:
            pitch = st.slider("Pitch", -10.0, 10.0, 0.0, 0.5, key="pitch")
    
    with model_tabs[2]:
        st.markdown("#### Analysis Provider")
        analysis_provider = st.radio(
            "Select Analysis Provider",
            ["default", "advanced"],
            key="analysis_provider",
            horizontal=True
        )
        
        st.markdown("#### Analysis Options")
        col1, col2 = st.columns(2)
        with col1:
            include_correlations = st.checkbox("Include Correlations", value=True, key="include_correlations")
        with col2:
            include_risk_metrics = st.checkbox("Include Risk Metrics", value=True, key="include_risk_metrics")
    
    # Collect all selections
    return {
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        "voice": {
            "stt_provider": stt_provider,
            "tts_provider": tts_provider,
            "voice": voice,
            "speaking_rate": speaking_rate,
            "pitch": pitch
        },
        "analysis": {
            "provider": analysis_provider,
            "include_correlations": include_correlations,
            "include_risk_metrics": include_risk_metrics
        }
    }


def render_analysis_dashboard(analysis_result):
    """
    Render a comprehensive financial analysis dashboard.
    
    Args:
        analysis_result: Analysis result from the analysis agent
    """
    if not analysis_result:
        st.warning("No analysis data available")
        return
    
    # Extract components
    exposures = analysis_result.get("exposures", {})
    changes = analysis_result.get("changes", {})
    volatility = analysis_result.get("volatility", {})
    correlations = analysis_result.get("correlations")
    risk_metrics = analysis_result.get("risk_metrics")
    summary = analysis_result.get("summary", "No summary available")
    provider_info = analysis_result.get("provider_info", {})
    
    # Create dashboard layout
    st.markdown("### Financial Analysis Dashboard")
    
    # Show provider info
    st.markdown(f"""
    <div class="provider-info">
        Analysis by: <b>{provider_info.get('name', 'Unknown')}</b> 
        (v{provider_info.get('version', '1.0.0')})
        {' ⚠️ Fallback provider used' if provider_info.get('fallback_used', False) else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Summary section
    st.markdown("#### Analysis Summary")
    st.markdown(f"<div class='analysis-summary'>{summary.replace('- ', '• ').replace('\\n', '<br>')}</div>", unsafe_allow_html=True)
    
    # Portfolio exposures
    if exposures:
        st.plotly_chart(render_portfolio_exposures(exposures), use_container_width=True)
    
    # Price changes and volatility
    col1, col2 = st.columns(2)
    
    with col1:
        if changes:
            st.plotly_chart(render_price_changes(changes), use_container_width=True)
    
    with col2:
        if volatility:
            st.plotly_chart(render_volatility_comparison(volatility), use_container_width=True)
    
    # Correlations
    if correlations:
        st.markdown("#### Asset Correlations")
        st.plotly_chart(render_correlation_matrix(correlations), use_container_width=True)
    
    # Risk metrics
    if risk_metrics:
        st.markdown("#### Risk Analysis")
        
        # Create tabs for each ticker
        tickers = list(risk_metrics.keys())
        if tickers:
            risk_tabs = st.tabs(tickers)
            
            for i, ticker in enumerate(tickers):
                with risk_tabs[i]:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        metrics = risk_metrics[ticker]
                        st.markdown(f"#### {ticker} Risk Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': [
                                'Sharpe Ratio',
                                'Sortino Ratio',
                                'Max Drawdown',
                                'Value at Risk (95%)',
                                'Conditional VaR (95%)',
                                'Beta',
                                'Calmar Ratio'
                            ],
                            'Value': [
                                f"{metrics.get('sharpe_ratio', 0):.2f}",
                                f"{metrics.get('sortino_ratio', 0):.2f}",
                                f"{metrics.get('max_drawdown', 0):.2%}",
                                f"{metrics.get('var_95', 0):.2%}",
                                f"{metrics.get('cvar_95', 0):.2%}" if 'cvar_95' in metrics else 'N/A',
                                f"{metrics.get('beta', 0):.2f}",
                                f"{metrics.get('calmar_ratio', 0):.2f}" if 'calmar_ratio' in metrics else 'N/A'
                            ]
                        })
                        st.table(metrics_df)
                    
                    with col2:
                        st.plotly_chart(render_risk_metrics_radar(risk_metrics, ticker), use_container_width=True)
