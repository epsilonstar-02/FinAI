import time
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime


def render_card(title, content, icon=None, card_type="default"):
    """
    Renders a styled card with title and content.
    
    Args:
        title: The card title
        content: The card content text
        icon: Optional icon name from FontAwesome
        card_type: Type of card (default, success, warning, error, info)
    """
    # Determine card class based on type
    card_class = f"card card-{card_type}"
    
    # Add icon if provided
    icon_html = f'<i class="fas fa-{icon} card-icon"></i>' if icon else ''
    
    # Markdown format the content
    content = content.replace('\n', '<br>')
    
    # Render timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    card_html = f"""
    <div class="{card_class}">
        <div class="card-header">
            {icon_html}
            <div class="card-title">{title}</div>
            <div class="card-timestamp">{timestamp}</div>
        </div>
        <div class="card-content">{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_audio_player(audio_bytes, autoplay=False):
    """
    Renders an audio player with the provided audio bytes.
    
    Args:
        audio_bytes: The audio data in bytes
        autoplay: Whether to autoplay the audio
    """
    if audio_bytes:
        st.markdown("<div class='audio-container'>", unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/mp3")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Auto-play script if enabled
        if autoplay:
            st.markdown("""
            <script>
                setTimeout(function() {
                    document.querySelector('audio').play();
                }, 1000);
            </script>
            """, unsafe_allow_html=True)


def show_progress_step(name, latency_ms):
    """
    Shows a progress step with a simulated delay.
    
    Args:
        name: The name of the step
        latency_ms: The simulated latency in milliseconds
    """
    start_time = time.time()
    with st.spinner(f"{name}..."):
        time.sleep(latency_ms / 1000.0)
    elapsed = time.time() - start_time
    st.markdown(f"<div class='step-complete'><i class='fas fa-check-circle'></i> {name} completed in {elapsed:.2f}s</div>", unsafe_allow_html=True)


def render_stock_info(ticker, price, change_pct):
    """
    Renders a stock info card with ticker, price and percent change.
    
    Args:
        ticker: Stock ticker symbol
        price: Current price
        change_pct: Percentage change
    """
    # Determine if stock is up or down
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
    """
    Renders a simple stock chart.
    
    Args:
        data: DataFrame with date and price columns
    """
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Create the chart
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('price:Q', title='Price ($)'),
        tooltip=['date', 'price']
    ).properties(
        width='container',
        height=300,
        title='Market Performance'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def render_tabs(tabs_content):
    """
    Renders custom styled tabs.
    
    Args:
        tabs_content: Dictionary of tab titles and their content functions
    """
    tab_titles = list(tabs_content.keys())
    tabs = st.tabs(tab_titles)
    
    for i, tab in enumerate(tabs):
        with tab:
            tabs_content[tab_titles[i]]()
