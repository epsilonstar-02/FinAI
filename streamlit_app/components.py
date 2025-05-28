import time
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime


def render_card(title, content, icon=None, card_type="default"):
    card_class = f"card card-{card_type}"
    icon_html = f'<i class="fas {icon} card-icon"></i>' if icon else ''
    
    import re
    content_html = content.replace('\n', '<br>')
    content_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content_html) # Bold
    content_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content_html)       # Italic
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    card_html = f"""
    <div class="{card_class}">
        <div class="card-header">
            {icon_html}
            <div class="card-title">{title}</div>
            <div class="card-timestamp">{timestamp}</div>
        </div>
        <div class="card-content">{content_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_audio_player(audio_data, autoplay=False, format="audio/mp3"):
    """
    Renders an audio player.
    Args:
        audio_data: Bytes of the audio file or a URL string.
        autoplay: Whether to autoplay the audio.
        format: The audio format string.
    """
    if audio_data:
        st.markdown("<div class='audio-player-container'>", unsafe_allow_html=True)
        st.audio(audio_data, format=format, start_time=0)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if autoplay:
            st.markdown("""
            <script>
                setTimeout(function() {
                    const audioElements = document.querySelectorAll('audio');
                    if (audioElements.length > 0) {
                        audioElements[audioElements.length - 1].play().catch(e => console.warn("Autoplay prevented: ", e));
                    }
                }, 500);
            </script>
            """, unsafe_allow_html=True)


def show_progress_step(name, latency_ms_simulated):
    """
    Shows a progress step.
    """
    st.markdown(f"<div class='step-complete'><i class='fas fa-check-circle'></i> {name} completed.</div>", unsafe_allow_html=True)


def render_stock_info(ticker, price, change_pct):
    change_class = "stock-up" if change_pct >= 0 else "stock-down"
    change_icon = "▲" if change_pct >= 0 else "▼"
    
    stock_html = f"""
    <div class="stock-card">
        <div class="stock-ticker">{ticker}</div>
        <div class="stock-price">${price:.2f}</div>
        <div class="stock-change {change_class}">
            {change_icon} {abs(change_pct):.2f}%
        </div>
    </div>
    """
    st.markdown(stock_html, unsafe_allow_html=True)


def render_market_chart(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    if 'date' not in data.columns or 'price' not in data.columns:
        st.warning("Market chart data requires 'date' and 'price' columns.")
        return

    data['date'] = pd.to_datetime(data['date'])

    chart = alt.Chart(data).mark_line(
        point=alt.OverlayMarkDef(size=20, filled=False, strokeWidth=1) # Color will be handled by theme
        # line=alt.LineConfig(strokeWidth=2) # Color will be handled by theme, or set explicitly below
    ).encode(
        x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %d', labelAngle=-45)),
        y=alt.Y('price:Q', title='Price ($)', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'), alt.Tooltip('price:Q', title='Price', format='$.2f')],
        color=alt.value('#2563eb') # Optionally, set a specific primary line color from your theme (e.g., light theme's primary blue: --primary-color)
    ).properties(
        height=300
    ).interactive()
    
    # Use theme="streamlit" for better integration with Streamlit's native light/dark themes
    st.altair_chart(chart, use_container_width=True, theme="streamlit")


def render_tabs(tabs_content: dict):
    tab_titles = list(tabs_content.keys())
    
    selected_tabs = st.tabs(tab_titles)
    
    for i, tab_title in enumerate(tab_titles):
        with selected_tabs[i]:
            tabs_content[tab_title]()