# streamlit_app/advanced_components.py
# Refinements for robustness, clarity, and handling missing data in Plotly charts.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional # Added Optional
import logging

logger = logging.getLogger(__name__)

# Helper from components.py, or define locally if preferred
def _format_value(value, precision=2, is_percent=False, default_na="N/A"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)): # Handle None and NaN/inf
        return default_na
    try:
        num = float(value)
        if is_percent:
            return f"{num:.{precision}%}" # Uses value directly as 0.xx for percent
        return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def render_provider_selector(section_title: str, providers: List[str], default_provider: Optional[str] = None, key_prefix: str = "") -> str:
    """
    Renders a selector for choosing among a list of providers.
    Uses session state to maintain selection.
    Returns the selected provider string.
    """
    st.subheader(section_title)
    
    if not providers:
        st.caption("No providers available for selection.")
        return "" # Or raise error, or return a sensible default

    cols = st.columns(len(providers))
    
    state_key = f"{key_prefix}_provider_selection"
    
    # Initialize session state if not already set, or if default_provider changed
    if state_key not in st.session_state or st.session_state[state_key] not in providers:
        if default_provider and default_provider in providers:
            st.session_state[state_key] = default_provider
        elif providers: # Default to the first provider if default is invalid or None
            st.session_state[state_key] = providers[0]
        else: # Should not happen if `if not providers:` check above is hit
             st.session_state[state_key] = "" 
    
    selected_provider = st.session_state[state_key]

    for i, provider_name in enumerate(providers):
        with cols[i]:
            is_selected = (selected_provider == provider_name)
            button_type = "primary" if is_selected else "secondary"
            button_key = f"{key_prefix}_provider_btn_{provider_name}"

            if st.button(provider_name, key=button_key, use_container_width=True, type=button_type):
                st.session_state[state_key] = provider_name
                selected_provider = provider_name # Update immediately for return
                st.rerun() # Rerun to reflect selection change visually and in logic
    
    return selected_provider


def render_correlation_matrix(correlation_data: Optional[Dict[str, Dict[str, float]]], title: str = "Asset Correlation Matrix") -> Optional[go.Figure]:
    if not correlation_data:
        st.caption("No correlation data provided.")
        return None
    
    try:
        df = pd.DataFrame(correlation_data)
        if df.empty or df.shape[0] != df.shape[1]: # Basic check for valid matrix
            st.warning("Invalid or empty correlation data for matrix.")
            return None

        fig = px.imshow(
            df,
            text_auto=".2f", # Format text on cells
            aspect="auto",   # Adjust aspect ratio
            color_continuous_scale="RdBu_r", # Red-Blue diverging, red for negative
            zmin=-1, zmax=1, # Correlation bounds
            title=title
        )
        fig.update_layout(
            width=None, # Use container width
            height=max(300, 50 * len(df.columns)), # Dynamic height
            coloraxis_colorbar=dict(title="Corr.", thicknessmode="pixels", thickness=15),
            title_x=0.5,
            xaxis_tickangle=-45 # Improve label readability
        )
        fig.update_traces(hovertemplate="Corr(%{x}, %{y}): %{z}<extra></extra>")
        return fig
    except Exception as e:
        logger.error(f"Error rendering correlation matrix: {e}", exc_info=True)
        st.error("Could not render correlation matrix.")
        return None


def render_risk_metrics_radar(risk_metrics_all: Dict[str, Dict[str, float]], ticker: str, title_suffix: str = "Risk Profile") -> Optional[go.Figure]:
    if ticker not in risk_metrics_all or not risk_metrics_all[ticker]:
        st.caption(f"No risk metrics available for {ticker}.")
        return None
    
    metrics = risk_metrics_all[ticker]
    
    # Normalize metrics for radar chart (0 to 1, where 1 is "better" or "target")
    # This requires domain knowledge for each metric.
    # Example normalizations (these are illustrative and need tuning):
    # Sharpe: Assume range -2 to 3. Map to 0-1. ( (val - (-2)) / (3 - (-2)) )
    # Sortino: Similar to Sharpe.
    # Max Drawdown: Lower is better. (1 - val) where val is 0-1.
    # VaR: Usually negative. Lower absolute value is better. (1 - abs(val)/TypicalMaxVaRAbs)
    # Beta: Target is often 1.0. (1 - abs(val - 1.0)/MaxDeviationFrom1)
    
    # Simplified display values, assuming higher is generally better for ratio, lower for risk
    # This normalization makes the radar chart interpretable where larger area = "better profile"
    # based on these assumptions.
    
    # Values closer to 1 are "better" or "more desirable" on this radar
    metrics_to_plot = {
        "Sharpe Ratio": min(max(metrics.get("sharpe_ratio", 0) / 3, -1), 1) * 0.5 + 0.5, # Scale to 0-1 assuming -3 to 3 range
        "Sortino Ratio": min(max(metrics.get("sortino_ratio", 0) / 4, -1), 1) * 0.5 + 0.5, # Scale to 0-1 assuming -4 to 4 range
        "Calmar Ratio": min(max(metrics.get("calmar_ratio", 0) / 5, -1), 1) * 0.5 + 0.5, # Scale
        "Max Drawdown": 1.0 - min(metrics.get("max_drawdown", 1.0), 1.0), # 0-1, lower is better, so 1-val
        "VaR (95%)": 1.0 - min(abs(metrics.get("var_95", 0.2)), 0.2) / 0.2, # Assumes VaR is negative, max expected |VaR| is 0.2 (20%)
        "CVaR (95%)": 1.0 - min(abs(metrics.get("cvar_95", 0.25)), 0.25) / 0.25,
        "Beta": 1.0 - min(abs(metrics.get("beta", 1.0) - 1.0), 1.0) # Deviation from 1, max dev 1
    }
    
    # Filter out metrics that are None from the original data
    valid_plot_metrics = {k: v for k, v in metrics_to_plot.items() if metrics.get(k.lower().replace(" (95%)","").replace(" ","_")) is not None}

    if not valid_plot_metrics:
        st.caption(f"Not enough valid risk metrics to plot radar for {ticker}.")
        return None

    categories = list(valid_plot_metrics.keys())
    values = [max(0, min(1, v)) for v in valid_plot_metrics.values()] # Ensure 0-1 range

    if not categories: # Should not happen if valid_plot_metrics is checked
        return None

    # Close the radar shape
    final_categories = categories + [categories[0]]
    final_values = values + [values[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=final_values,
        theta=final_categories,
        fill='toself',
        name=ticker,
        line_color='var(--primary-color)', # Use CSS variable (Plotly might not support this directly in all contexts)
                                          # Fallback: 'rgba(37, 99, 235, 0.9)' 
        fillcolor='rgba(37, 99, 235, 0.3)' # Example color
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=9),
                   angularaxis=dict(tickfont_size=10, direction="clockwise")),
        showlegend=False, title=f"{ticker} - {title_suffix}", title_x=0.5,
        height=350, margin=dict(l=30, r=30, t=70, b=30) # Adjusted margins
    )
    return fig


def render_portfolio_exposures(exposures: Optional[Dict[str, float]], title: str = "Portfolio Allocation") -> Optional[go.Figure]:
    if not exposures or not any(v > 0 for v in exposures.values()): # Check if any exposure is positive
        st.caption("No portfolio exposure data to display.")
        return None

    # Filter out zero or negative exposures for pie chart
    valid_exposures = {k: v for k, v in exposures.items() if v > 1e-6} # Threshold for positive
    if not valid_exposures:
        st.caption("All portfolio exposures are zero or negligible.")
        return None

    labels = list(valid_exposures.keys())
    values = [valid_exposures[symbol] * 100 for symbol in labels] # Convert to percentage
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.4,
        textinfo='label+percent', marker=dict(colors=px.colors.qualitative.Plotly), # Standard color sequence
        hoverinfo='label+percent+value', textfont_size=11, pull=[0.05 if v == max(values) else 0 for v in values] # Pull largest slice
    )])
    fig.update_layout(
        title=title, title_x=0.5, height=380,
        margin=dict(t=60, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def render_price_changes(changes: Optional[Dict[str, float]], title: str = "Daily Price Changes (%)") -> Optional[go.Figure]:
    if not changes:
        st.caption("No price change data to display.")
        return None

    # Filter out None or non-finite values
    valid_changes = {sym: val for sym, val in changes.items() if val is not None and np.isfinite(val)}
    if not valid_changes:
        st.caption("Price change data contains no valid numeric values.")
        return None

    symbols = list(valid_changes.keys())
    pct_changes = [valid_changes[symbol] * 100 for symbol in symbols] # Convert to percentage
    
    # Sort by symbol name for consistent display, or by change value
    sorted_data = sorted(zip(symbols, pct_changes), key=lambda item: item[0])
    symbols = [item[0] for item in sorted_data]
    pct_changes = [item[1] for item in sorted_data]
    
    colors = ['var(--stock-up-color)' if x >= 0 else 'var(--stock-down-color)' for x in pct_changes]
    # Fallback colors if CSS vars don't work in Plotly:
    # colors = ['#10b981' if x >= 0 else '#ef4444' for x in pct_changes]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=symbols, x=pct_changes, orientation='h',
        marker_color=colors, 
        text=[f"{_format_value(x, 2)}%" for x in pct_changes], # Use helper
        textposition='auto', insidetextanchor='middle', textfont_size=10
    ))
    fig.update_layout(
        title=title, title_x=0.5, xaxis_title="Percent Change",
        yaxis=dict(tickfont=dict(size=10), automargin=True), # automargin for long labels
        height=max(200, len(symbols) * 30 + 80), # Dynamic height
        margin=dict(t=50, b=40, l=10, r=20, pad=4) # Pad for text
    )
    return fig


def render_volatility_comparison(volatility: Optional[Dict[str, float]], title: str = "Asset Volatility Comparison (%)") -> Optional[go.Figure]:
    if not volatility:
        st.caption("No volatility data to display.")
        return None

    valid_volatility = {sym: val for sym, val in volatility.items() if val is not None and np.isfinite(val)}
    if not valid_volatility:
        st.caption("Volatility data contains no valid numeric values.")
        return None

    symbols = list(valid_volatility.keys())
    vol_values = [valid_volatility[symbol] * 100 for symbol in symbols] # Annualize or show as is? Assuming daily shown as %
    
    sorted_data = sorted(zip(symbols, vol_values), key=lambda item: item[1], reverse=True) # Sort by vol desc
    symbols = [item[0] for item in sorted_data]
    vol_values = [item[1] for item in sorted_data]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols, y=vol_values,
        marker_color='var(--secondary-color)', # Use CSS var, fallback: 'rgba(15, 118, 110, 0.8)'
        text=[f"{_format_value(x, 2)}%" for x in vol_values],
        textposition='auto', textfont_size=10
    ))
    fig.update_layout(
        title=title, title_x=0.5, yaxis_title="Volatility (%)",
        xaxis=dict(tickfont=dict(size=10)),
        height=max(300, len(symbols) * 20 + 100), # Dynamic height
        margin=dict(t=50, b=40, l=10, r=10)
    )
    return fig


def render_model_selector(show_subheader: bool = True) -> Dict[str, Dict[str, Any]]:
    if show_subheader:
        st.subheader("🤖 Model & Analysis Configuration") # More descriptive

    # Use session state for defaults to persist user choices across reruns
    # Initialize if not present
    if "model_selector_state" not in st.session_state:
        st.session_state.model_selector_state = {
            "llm_provider": "Gemini", "llm_model": "gemini-flash",
            "temperature": 0.7, "max_tokens": 1000,
            "stt_provider": "Whisper", "tts_provider": "Google TTS",
            "voice": "en-US-Neural2-F", "speaking_rate": 1.0, "pitch": 0.0,
            "analysis_provider": "advanced", # Default to advanced for more features
            "include_correlations": True, "include_risk_metrics": True
        }
    
    # Helper to get value from session state or default if key missing (for robustness)
    def _get_state_val(key, default_val):
        return st.session_state.model_selector_state.get(key, default_val)

    s = st.session_state.model_selector_state # Shorthand

    model_tabs = st.tabs(["💬 LLM Settings", "🎤 Voice Settings", "📊 Analysis Settings"])
    
    with model_tabs[0]: # LLM
        st.markdown("##### Language Model (LLM)")
        col1, col2 = st.columns(2)
        with col1:
            s["llm_provider"] = st.selectbox(
                "Provider", ["Gemini", "OpenAI", "Anthropic", "HuggingFace"], # TODO: Add Llama if local option in Streamlit is feasible
                index=["Gemini", "OpenAI", "Anthropic", "HuggingFace"].index(_get_state_val("llm_provider","Gemini")),
                key="llm_provider_selector_adv" 
            )
        with col2:
            if s["llm_provider"] == "Gemini":
                s["llm_model"] = st.selectbox("Model", ["gemini-flash", "gemini-pro", "gemini-1.5-pro"], index=["gemini-flash", "gemini-pro", "gemini-1.5-pro"].index(_get_state_val("llm_model","gemini-flash")), key="gemini_model_selector_adv")
            elif s["llm_provider"] == "OpenAI":
                s["llm_model"] = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"].index(_get_state_val("llm_model","gpt-3.5-turbo")), key="openai_model_selector_adv")
            # Add Anthropic, HuggingFace models similarly
            elif s["llm_provider"] == "Anthropic":
                 s["llm_model"] = st.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"], index=0, key="anthropic_model_selector_adv") # Default to first
            else: # HuggingFace
                 s["llm_model"] = st.text_input("Model Name/Path (HuggingFace)", value=_get_state_val("llm_model","mistralai/Mistral-7B-Instruct-v0.2"), key="hf_model_selector_adv")
        
        st.markdown("###### LLM Parameters")
        s["temperature"] = st.slider("Temperature", 0.0, 1.0, _get_state_val("temperature",0.7), 0.05, key="temperature_slider_adv", help="Lower for more factual, higher for more creative.")
        s["max_tokens"] = st.slider("Max Output Tokens", 100, 4096, _get_state_val("max_tokens",1000), 50, key="max_tokens_slider_adv", help="Max number of tokens in generated response.")
    
    with model_tabs[1]: # Voice
        st.markdown("##### Speech-to-Text (STT)")
        # TODO: Populate STT/TTS providers dynamically from Voice Agent's /providers endpoint if possible
        s["stt_provider"] = st.selectbox("STT Provider", ["Whisper", "Google Cloud STT", "Azure Speech"], index=0, key="stt_provider_selector_adv") # Example providers

        st.markdown("##### Text-to-Speech (TTS)")
        s["tts_provider"] = st.selectbox("TTS Provider", ["Google TTS", "Edge TTS", "ElevenLabs"], index=0, key="tts_provider_selector_adv") # Example providers
        
        s["voice"] = st.text_input("TTS Voice Accent/ID", value=_get_state_val("voice","en-US-Neural2-F"), key="voice_selector_adv", help="E.g., en-US-Neural2-F (Google), or specific ID for ElevenLabs.")
        col1_tts, col2_tts = st.columns(2)
        with col1_tts:
            s["speaking_rate"] = st.slider("Speaking Rate", 0.5, 2.0, _get_state_val("speaking_rate",1.0), 0.05, key="speaking_rate_slider_adv")
        with col2_tts:
            s["pitch"] = st.slider("Pitch Adjustment", -10.0, 10.0, _get_state_val("pitch",0.0), 0.5, key="pitch_slider_adv", help="Range varies by TTS provider.")

    with model_tabs[2]: # Analysis
        st.markdown("##### Financial Analysis Engine")
        s["analysis_provider"] = st.radio(
            "Analysis Provider", ["default", "advanced"], 
            index=["default", "advanced"].index(_get_state_val("analysis_provider","advanced")),
            key="analysis_provider_selector_adv", horizontal=True,
            help="'Default' for basic metrics, 'Advanced' for more detailed analysis."
        )
        st.markdown("###### Analysis Options")
        col1_an, col2_an = st.columns(2)
        with col1_an:
            s["include_correlations"] = st.checkbox("Include Correlations", value=_get_state_val("include_correlations",True), key="include_correlations_cb_adv")
        with col2_an:
            s["include_risk_metrics"] = st.checkbox("Include Risk Metrics", value=_get_state_val("include_risk_metrics",True), key="include_risk_metrics_cb_adv")
    
    # Return a structured dictionary matching the needs of app.py's orchestrator_params
    return {
        "llm_config": { # Changed key from "llm" for clarity
            "provider": s["llm_provider"], "model": s["llm_model"],
            "temperature": s["temperature"], "max_tokens": s["max_tokens"]
        },
        "voice_config": { # Changed key from "voice"
            "stt_provider": s["stt_provider"], "tts_provider": s["tts_provider"],
            "voice": s["voice"], "speaking_rate": s["speaking_rate"], "pitch": s["pitch"]
        },
        "analysis_config": { # Changed key from "analysis"
            "provider": s["analysis_provider"],
            "include_correlations": s["include_correlations"],
            "include_risk_metrics": s["include_risk_metrics"]
        }
    }


def render_analysis_dashboard(analysis_result: Optional[Dict[str, Any]]):
    if not analysis_result:
        st.info("No analysis data available to display. Please generate an analysis using the assistant.")
        return
    
    # Extract data with .get() for safety
    exposures = analysis_result.get("exposures")
    changes = analysis_result.get("changes")
    volatility = analysis_result.get("volatility")
    correlations = analysis_result.get("correlations")
    risk_metrics_all = analysis_result.get("risk_metrics") # Dict of dicts
    summary = analysis_result.get("summary", "No summary provided.")
    provider_info = analysis_result.get("provider_info", {})
    
    st.markdown("### Financial Analysis Dashboard")
    
    if provider_info:
        provider_name = provider_info.get('name', 'Unknown Provider')
        provider_version = provider_info.get('version', 'N/A')
        fallback_info = " (Fallback Used)" if provider_info.get('fallback_used', False) else ""
        exec_time = provider_info.get('execution_time_ms')
        exec_time_str = f", Exec Time: {_format_value(exec_time,0)}ms" if exec_time is not None else ""

        st.markdown(f"""
        <div class="provider-info">
            Analyzed by: <b>{provider_name}</b> (v{provider_version}){fallback_info}{exec_time_str}
        </div>""", unsafe_allow_html=True)
    
    st.markdown("#### Executive Summary")
    # Basic formatting for summary text
    summary_html = summary.replace('\n', '<br>').replace('- ', '• ')
    st.markdown(f"<div class='analysis-summary'>{summary_html}</div>", unsafe_allow_html=True)
    
    # Layout for charts
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if exposures: fig_exp = render_portfolio_exposures(exposures); st.plotly_chart(fig_exp, use_container_width=True) if fig_exp else None
        else: st.caption("Exposure data unavailable.")
    with row1_col2:
        if changes: fig_chg = render_price_changes(changes); st.plotly_chart(fig_chg, use_container_width=True) if fig_chg else None
        else: st.caption("Price change data unavailable.")

    if volatility: fig_vol = render_volatility_comparison(volatility); st.plotly_chart(fig_vol, use_container_width=True) if fig_vol else None
    else: st.caption("Volatility data unavailable.")
            
    if correlations: fig_corr = render_correlation_matrix(correlations); st.plotly_chart(fig_corr, use_container_width=True) if fig_corr else None
    else: st.caption("Correlation data unavailable.")
    
    # Risk Metrics Section
    if risk_metrics_all:
        st.markdown("#### Risk Analysis")
        tickers_with_risk = [sym for sym, metrics in risk_metrics_all.items() if metrics] # Filter out symbols with no/empty risk metrics
        
        if not tickers_with_risk:
            st.caption("No valid risk metrics data found for any ticker.")
            return

        if len(tickers_with_risk) > 3: # Use selectbox for many tickers
            selected_ticker = st.selectbox("Select Ticker for Risk Details", tickers_with_risk, key="risk_ticker_selector_adv")
            display_tickers = [selected_ticker] if selected_ticker else []
        else: # Use tabs for few tickers
            risk_tabs = st.tabs(tickers_with_risk)
            display_tickers = tickers_with_risk
        
        for i, ticker_key in enumerate(display_tickers):
            current_tab_context = risk_tabs[i] if len(tickers_with_risk) <= 3 and len(display_tickers) > 1 else st
            
            with current_tab_context:
                if not (len(tickers_with_risk) > 3) and len(display_tickers) > 1 : st.markdown(f"##### {ticker_key} Risk Profile") # Add sub-header in tabs

                metrics_data_for_ticker = risk_metrics_all.get(ticker_key)
                if not metrics_data_for_ticker: st.warning(f"No risk metrics found for {ticker_key}."); continue

                col_risk_table, col_risk_radar = st.columns([0.4, 0.6]) # Adjust column ratio
                with col_risk_table:
                    st.markdown(f"###### Key Metrics") # Removed ticker from here as it's in tab/title
                    
                    # More robust display of metrics, handling None or missing values
                    displayable_metrics = {
                        'Sharpe Ratio': _format_value(metrics_data_for_ticker.get('sharpe_ratio')),
                        'Sortino Ratio': _format_value(metrics_data_for_ticker.get('sortino_ratio')),
                        'Max Drawdown': _format_value(metrics_data_for_ticker.get('max_drawdown'), 2, is_percent=True),
                        'VaR (95%)': _format_value(metrics_data_for_ticker.get('var_95'), 2, is_percent=True),
                        'CVaR (95%)': _format_value(metrics_data_for_ticker.get('cvar_95'), 2, is_percent=True),
                        'Beta': _format_value(metrics_data_for_ticker.get('beta')),
                        'Calmar Ratio': _format_value(metrics_data_for_ticker.get('calmar_ratio'))
                    }
                    metrics_df = pd.DataFrame(displayable_metrics.items(), columns=['Metric', 'Value'])
                    st.table(metrics_df.set_index('Metric'))
                
                with col_risk_radar:
                    fig_radar = render_risk_metrics_radar(risk_metrics_all, ticker_key) # Pass all metrics, function will pick by ticker
                    if fig_radar: st.plotly_chart(fig_radar, use_container_width=True)
                    else: st.caption(f"Radar chart not available for {ticker_key}.")
    else:
        st.caption("Risk metrics data unavailable.")