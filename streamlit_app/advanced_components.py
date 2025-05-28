import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def render_provider_selector(section_title, providers, default=None, key_prefix=""):
    st.subheader(section_title)
    
    cols = st.columns(len(providers))
    
    state_key = f"{key_prefix}_provider_selection"
    if state_key not in st.session_state:
        st.session_state[state_key] = default if default in providers else providers[0]
    
    for i, provider in enumerate(providers):
        with cols[i]:
            is_selected = (st.session_state[state_key] == provider)
            button_type = "primary" if is_selected else "secondary"
            
            button_key = f"{key_prefix}_provider_btn_{provider}"

            if st.button(provider, key=button_key, use_container_width=True, type=button_type):
                st.session_state[state_key] = provider
                st.rerun()
    
    return st.session_state[state_key]


def render_correlation_matrix(correlation_data):
    df = pd.DataFrame(correlation_data)
    
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
            thickness=15,
            lenmode="pixels", 
            len=400,
        ),
        title_x=0.5
    )
    
    return fig


def render_risk_metrics_radar(risk_metrics, ticker):
    if ticker not in risk_metrics:
        st.warning(f"No risk metrics available for {ticker}.")
        return None
    
    metrics = risk_metrics[ticker]
    
    metrics_to_plot = {
        "Sharpe Ratio": min(max(metrics.get("sharpe_ratio", 0), -1) / 3, 1),
        "Sortino Ratio": min(max(metrics.get("sortino_ratio", 0), -1) / 3, 1),
        "1 - Max Drawdown": 1 - min(abs(metrics.get("max_drawdown", 0)), 1),
        "1 - VaR (95%)": 1 - min(abs(metrics.get("var_95", 0.2)), 1) / 1,
        "Beta": 1 - min(abs(metrics.get("beta", 1.0) - 1.0), 1.0) / 1.0
    }
    
    categories = list(metrics_to_plot.keys())
    values = list(metrics_to_plot.values())
    
    values = [max(0, min(1, v)) for v in values]

    if categories:
        categories.append(categories[0])
        values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=ticker,
        line_color='rgba(37, 99, 235, 0.9)', 
        fillcolor='rgba(37, 99, 235, 0.4)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title=f"Risk Profile: {ticker}",
        title_x=0.5,
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def render_portfolio_exposures(exposures):
    labels = list(exposures.keys())
    values = [exposures[symbol] * 100 for symbol in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.Plotly),
        hoverinfo='label+percent+value',
        textfont_size=11
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        title_x=0.5,
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def render_price_changes(changes):
    symbols = list(changes.keys())
    pct_changes = [changes[symbol] * 100 for symbol in symbols]
    
    sorted_indices = np.argsort(pct_changes)
    symbols = [symbols[i] for i in sorted_indices]
    pct_changes = [pct_changes[i] for i in sorted_indices]
    
    # Assuming CSS variables --stock-up-color and --stock-down-color are defined and accessible
    # If Plotly cannot access CSS variables directly, these should be replaced with actual hex/rgba values
    colors = ['#10b981' if x >= 0 else '#ef4444' for x in pct_changes] # Example: using hex from your CSS
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=symbols,
        x=pct_changes,
        orientation='h',
        marker_color=colors, 
        text=[f"{x:.2f}%" for x in pct_changes],
        textposition='auto',
        insidetextanchor='middle' if all(abs(x) > 5 for x in pct_changes) else 'auto',
        textfont_size=10
    ))
    
    fig.update_layout(
        title="Daily Price Changes",
        title_x=0.5,
        xaxis_title="Percent Change",
        yaxis=dict(tickfont=dict(size=10)),
        height=380,
        margin=dict(t=50, b=40, l=10, r=10)
    )
    
    return fig


def render_volatility_comparison(volatility):
    symbols = list(volatility.keys())
    vol_values = [volatility[symbol] * 100 for symbol in symbols]
    
    sorted_indices = np.argsort(vol_values)[::-1]
    symbols = [symbols[i] for i in sorted_indices]
    vol_values = [vol_values[i] for i in sorted_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols,
        y=vol_values,
        marker_color='rgba(15, 118, 110, 0.8)', 
        text=[f"{x:.2f}%" for x in vol_values],
        textposition='auto',
        textfont_size=10
    ))
    
    fig.update_layout(
        title="Asset Volatility Comparison",
        title_x=0.5,
        yaxis_title="Volatility (%)",
        xaxis=dict(tickfont=dict(size=10)),
        height=380,
        margin=dict(t=50, b=40, l=10, r=10)
    )
    
    return fig


def render_model_selector(show_subheader=True): # Added show_subheader parameter
    if show_subheader:
        st.subheader("Model Selection") # Controlled by the new parameter
    
    model_tabs = st.tabs(["💬 LLM", "🎤 Voice", "📊 Analyze"])
    
    # Initialize variables to ensure they are always defined
    llm_provider = "Gemini" # Default provider
    llm_model = "gemini-flash" # Default model for the default provider
    temperature = 0.7
    max_tokens = 1000
    stt_provider = "Whisper"
    tts_provider = "Google TTS"
    voice = "en-US-Neural2-F"
    speaking_rate = 1.0
    pitch = 0.0
    analysis_provider = "default"
    include_correlations = True
    include_risk_metrics = True

    with model_tabs[0]: 
        st.markdown("##### LLM Configuration") 
        col1, col2 = st.columns(2)
        with col1:
            llm_provider = st.selectbox(
                "LLM Provider", 
                ["Gemini", "OpenAI", "Anthropic", "HuggingFace"],
                key="llm_provider_selector" 
            )
        
        with col2:
            if llm_provider == "Gemini":
                llm_model = st.selectbox("Model", ["gemini-flash", "gemini-pro", "gemini-1.5-pro"], key="gemini_model_selector")
            elif llm_provider == "OpenAI":
                llm_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], key="openai_model_selector")
            elif llm_provider == "Anthropic":
                llm_model = st.selectbox("Model", ["claude-2", "claude-instant", "claude-3-opus"], key="anthropic_model_selector")
            else: 
                llm_model = st.selectbox("Model", ["llama-2-7b", "mistral-7b", "falcon-7b"], key="hf_model_selector")
        
        st.markdown("###### Parameters") 
        col1_params, col2_params = st.columns(2)
        with col1_params:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature_slider")
        with col2_params:
            max_tokens = st.slider("Max Tokens", 100, 4000, 1000, 100, key="max_tokens_slider")
    
    with model_tabs[1]: 
        st.markdown("##### Voice Agent Configuration") 
        col1_voice, col2_voice = st.columns(2)
        with col1_voice:
            stt_provider = st.selectbox("STT Provider", ["Whisper", "Google Cloud", "Azure", "Vosk"], key="stt_provider_selector")
        with col2_voice:
            tts_provider = st.selectbox("TTS Provider", ["Google TTS", "Edge TTS", "ElevenLabs", "Amazon Polly"], key="tts_provider_selector")
        
        st.markdown("###### TTS Parameters") 
        col1_tts, col2_tts, col3_tts = st.columns(3)
        with col1_tts:
            voice = st.selectbox("Voice Accent", ["en-US-Neural2-F", "en-US-Neural2-M", "en-GB-Neural2-F", "en-GB-Neural2-M"], key="voice_selector")
        with col2_tts:
            speaking_rate = st.slider("Speaking Rate", 0.5, 2.0, 1.0, 0.1, key="speaking_rate_slider")
        with col3_tts:
            pitch = st.slider("Pitch", -10.0, 10.0, 0.0, 0.5, key="pitch_slider")
    
    with model_tabs[2]: 
        st.markdown("##### Analysis Engine") 
        analysis_provider = st.radio(
            "Provider", 
            ["default", "advanced"],
            key="analysis_provider_selector",
            horizontal=True,
            label_visibility="collapsed" 
        )
        
        st.markdown("###### Options") 
        col1_analysis, col2_analysis = st.columns(2)
        with col1_analysis:
            include_correlations = st.checkbox("Correlations", value=True, key="include_correlations_cb")
        with col2_analysis:
            include_risk_metrics = st.checkbox("Risk Metrics", value=True, key="include_risk_metrics_cb")
    
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
    if not analysis_result:
        st.info("No analysis data available to display.")
        return
    
    exposures = analysis_result.get("exposures", {})
    changes = analysis_result.get("changes", {})
    volatility = analysis_result.get("volatility", {})
    correlations = analysis_result.get("correlations")
    risk_metrics = analysis_result.get("risk_metrics")
    summary = analysis_result.get("summary", "No summary available.")
    provider_info = analysis_result.get("provider_info", {})
    
    st.markdown("### Financial Analysis Dashboard")
    
    if provider_info:
        st.markdown(f"""
        <div class="provider-info" style="font-size: 0.9em; color: var(--text-tertiary); margin-bottom: 1rem;">
            Analysis by: <b>{provider_info.get('name', 'Unknown')}</b> 
            (v{provider_info.get('version', 'N/A')})
            {' <span style="color: var(--warning-color);">⚠️ Fallback provider used</span>' if provider_info.get('fallback_used', False) else ''}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### Executive Summary")
    formatted_summary = summary.replace('- ', '• ').replace('\n', '<br>')
    st.markdown(f"<div class='analysis-summary' style='background-color: var(--light-gray); padding: 1rem; border-radius: var(--radius-md); margin-bottom: 1.5rem;'>{formatted_summary}</div>", unsafe_allow_html=True)
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        if exposures:
            st.plotly_chart(render_portfolio_exposures(exposures), use_container_width=True)
        else:
            st.caption("No exposure data.")
    
    with row1_col2:
        if changes:
            st.plotly_chart(render_price_changes(changes), use_container_width=True)
        else:
            st.caption("No price change data.")

    if volatility:
        st.plotly_chart(render_volatility_comparison(volatility), use_container_width=True)
    else:
        st.caption("No volatility data.")
            
    if correlations:
        st.markdown("#### Asset Correlations")
        st.plotly_chart(render_correlation_matrix(correlations), use_container_width=True)
    else:
        st.caption("No correlation data.")
    
    if risk_metrics:
        st.markdown("#### Risk Analysis")
        tickers = list(risk_metrics.keys())
        if tickers:
            risk_tabs_ctx = None # Initialize to avoid UnboundLocalError
            if len(tickers) > 3:
                selected_ticker_for_risk = st.selectbox("Select Ticker for Risk Details", tickers, key="risk_ticker_selector")
                tickers_to_display = [selected_ticker_for_risk]
            else:
                risk_tabs = st.tabs(tickers)
                tickers_to_display = tickers
                risk_tabs_ctx = risk_tabs 

            for i, ticker_key in enumerate(tickers_to_display):
                current_display_context = st 
                if not (len(tickers) > 3) and risk_tabs_ctx: 
                    current_display_context = risk_tabs_ctx[i]

                with current_display_context:
                    if not (len(tickers) > 3) and len(tickers_to_display) > 1 : 
                        st.markdown(f"##### {ticker_key} Risk Profile")

                    metrics_data = risk_metrics.get(ticker_key)
                    if not metrics_data:
                        st.warning(f"No risk metrics found for {ticker_key}")
                        continue

                    col_risk_table, col_risk_radar = st.columns([1, 1])
                    with col_risk_table:
                        st.markdown(f"###### Key Metrics: {ticker_key}")
                        
                        available_metrics = {
                            'Sharpe Ratio': f"{metrics_data.get('sharpe_ratio', 'N/A'):.2f}" if isinstance(metrics_data.get('sharpe_ratio'), (int, float)) else 'N/A',
                            'Sortino Ratio': f"{metrics_data.get('sortino_ratio', 'N/A'):.2f}" if isinstance(metrics_data.get('sortino_ratio'), (int, float)) else 'N/A',
                            'Max Drawdown': f"{metrics_data.get('max_drawdown', 'N/A'):.2%}" if isinstance(metrics_data.get('max_drawdown'), (int, float)) else 'N/A',
                            'VaR (95%)': f"{metrics_data.get('var_95', 'N/A'):.2%}" if isinstance(metrics_data.get('var_95'), (int, float)) else 'N/A',
                            'CVaR (95%)': f"{metrics_data.get('cvar_95', 'N/A'):.2%}" if isinstance(metrics_data.get('cvar_95'), (int, float)) else 'N/A',
                            'Beta': f"{metrics_data.get('beta', 'N/A'):.2f}" if isinstance(metrics_data.get('beta'), (int, float)) else 'N/A',
                            'Calmar Ratio': f"{metrics_data.get('calmar_ratio', 'N/A'):.2f}" if isinstance(metrics_data.get('calmar_ratio'), (int, float)) else 'N/A'
                        }
                        
                        metrics_df_data = {'Metric': [], 'Value': []}
                        for m, v in available_metrics.items():
                            metrics_df_data['Metric'].append(m)
                            metrics_df_data['Value'].append(v)
                        
                        metrics_df = pd.DataFrame(metrics_df_data)
                        st.table(metrics_df.set_index('Metric'))
                    
                    with col_risk_radar:
                        radar_fig = render_risk_metrics_radar(risk_metrics, ticker_key)
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                        else:
                            st.caption(f"Radar chart not available for {ticker_key}.")
        else:
            st.caption("No tickers found for risk metrics.")
    else:
        st.caption("No risk metrics data.")