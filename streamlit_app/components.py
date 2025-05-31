# streamlit_app/components.py
# Refinements for clarity, robustness, and minor enhancements.

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import re # For markdown-like parsing in render_card

# Helper for formatting numbers, can be expanded
def _format_value(value, precision=2, is_percent=False):
    if value is None:
        return "N/A"
    try:
        num = float(value)
        if is_percent:
            return f"{num:.{precision}%}"
        return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return str(value) # Fallback for non-numeric

def render_card(title: str, content: str, icon: str = "fas fa-info-circle", card_type: str = "default", timestamp_override: Optional[str] = None):
    """
    Renders a styled card with a title, content, and an optional icon.
    card_type can be "default", "success", "warning", "error", "info".
    """
    card_class = f"card card-{card_type}" # CSS classes from styles.css
    icon_html = f'<i class="{icon} card-icon"></i>' if icon else ''
    
    # Basic markdown-to-HTML for content (bold, italic, lists)
    # Replace **text** with <strong>text</strong>
    content_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    # Replace *text* or _text_ with <em>text</em>
    content_html = re.sub(r'(?<!\*)\*(?!\s|\*)(.*?)(?<!\s|\*)\*(?!\*)', r'<em>\1</em>', content_html)
    content_html = re.sub(r'(?<!_)_(?!\s|_)(.*?)(?<!\s|_)_(?!_)', r'<em>\1</em>', content_html)
    
    # Handle simple unordered lists like "- item" or "* item"
    list_items_html = []
    is_in_list = False
    for line in content_html.split('\n'):
        if line.strip().startswith(("- ", "* ")):
            if not is_in_list:
                list_items_html.append("<ul>")
                is_in_list = True
            list_items_html.append(f"<li>{line.strip()[2:]}</li>")
        else:
            if is_in_list:
                list_items_html.append("</ul>")
                is_in_list = False
            list_items_html.append(line + "<br>") # Keep line breaks for non-list lines
    if is_in_list: # Close list if content ends with list items
        list_items_html.append("</ul>")
    
    content_html = "".join(list_items_html)
    # Remove trailing <br> if it's the last element after ul
    if content_html.endswith("</ul><br>"):
        content_html = content_html[:-4] + "</ul>"
    elif content_html.endswith("<br>"):
         content_html = content_html[:-4]


    timestamp_str = timestamp_override if timestamp_override else datetime.now().strftime("%H:%M:%S")
    
    card_html = f"""
    <div class="{card_class}">
        <div class="card-header">
            {icon_html}
            <div class="card-title">{title}</div>
            <div class="card-timestamp">{timestamp_str}</div>
        </div>
        <div class="card-content">{content_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_audio_player(audio_data: bytes, autoplay: bool = False, format_type: str = "audio/mp3"):
    """
    Renders an HTML5 audio player for the given audio data.
    Autoplay is subject to browser restrictions.
    """
    if not audio_data:
        st.warning("No audio data provided to player.")
        return

    st.markdown("<div class='audio-player-container'>", unsafe_allow_html=True)
    st.audio(audio_data, format=format_type, start_time=0)
    st.markdown("</div>", unsafe_allow_html=True)
        
    if autoplay:
        # Unique ID for this audio player to target specifically
        player_id = f"audio_player_{hash(audio_data)}" # Simple hash for some uniqueness
        
        # This JS tries to find the *last* audio element added by st.audio
        # A more robust way would be if st.audio returned an ID or allowed setting one.
        # For now, assuming it's the last one for autoplay.
        # Browsers often block autoplay unless user has interacted with the page.
        autoplay_script = f"""
            <script>
                setTimeout(function() {{
                    const audioElements = document.querySelectorAll('audio');
                    if (audioElements.length > 0) {{
                        const lastPlayer = audioElements[audioElements.length - 1];
                        lastPlayer.play().catch(e => {{
                            console.warn("Autoplay was prevented by the browser: ", e.message);
                            // Optionally, show a 'Click to play' button if autoplay fails
                        }});
                    }}
                }}, 500); // Delay to ensure element is in DOM
            </script>
        """
        st.markdown(autoplay_script, unsafe_allow_html=True)


def show_progress_step(step_name: str, is_complete: bool = True, details: Optional[str] = None):
    """
    Displays a simulated progress step.
    If used for actual steps, `is_complete` would be dynamic.
    """
    icon_class = "fa-check-circle" if is_complete else "fa-spinner fa-spin"
    status_class = "step-complete" if is_complete else "step-in-progress" # Define .step-in-progress in CSS
    
    step_html = f"""
    <div class='{status_class}'>
        <i class='fas {icon_class}'></i> {step_name}
        {f"<small style='margin-left:10px; color: var(--text-tertiary);'>({details})</small>" if details else ""}
    </div>
    """
    st.markdown(step_html, unsafe_allow_html=True)


def render_stock_info(ticker: str, price: Optional[float], change_pct: Optional[float]):
    """Renders a card-like display for a single stock's info."""
    price_str = _format_value(price, 2)
    change_str = "N/A"
    change_class = "stock-neutral" # Default class
    change_icon = "●" # Neutral icon

    if change_pct is not None:
        change_str = f"{_format_value(abs(change_pct), 2)}%"
        if change_pct > 0:
            change_class = "stock-up"
            change_icon = "▲"
        elif change_pct < 0:
            change_class = "stock-down"
            change_icon = "▼"
            
    stock_html = f"""
    <div class="stock-card">
        <div class="stock-ticker">{ticker.upper()}</div>
        <div class="stock-price">${price_str}</div>
        <div class="stock-change {change_class}">
            {change_icon} {change_str}
        </div>
    </div>
    """
    st.markdown(stock_html, unsafe_allow_html=True)


def render_market_chart(data: pd.DataFrame, x_col: str = 'date', y_col: str = 'price', 
                        x_title: str = 'Date', y_title: str = 'Price ($)', chart_height: int = 300):
    """
    Renders a line chart for market data using Altair.
    Data DataFrame must contain x_col and y_col.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.caption("No data available for market chart.")
        return
    
    if x_col not in data.columns or y_col not in data.columns:
        st.warning(f"Market chart data requires '{x_col}' and '{y_col}' columns. Found: {list(data.columns)}")
        return

    # Ensure x_col is datetime for proper Altair time axis
    try:
        data[x_col] = pd.to_datetime(data[x_col])
    except Exception as e:
        st.warning(f"Could not convert '{x_col}' to datetime for chart: {e}. Chart may not render correctly.")
        # Attempt to plot anyway, Altair might handle some string dates.

    try:
        data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
        data = data.dropna(subset=[y_col]) # Drop rows where y_col could not be converted
        if data.empty:
            st.caption(f"No valid numeric data in '{y_col}' for market chart after conversion.")
            return
    except Exception as e:
        st.warning(f"Could not convert '{y_col}' to numeric for chart: {e}. Chart may not render correctly.")


    chart = alt.Chart(data).mark_line(
        point=alt.OverlayMarkDef(size=30, filled=False, strokeWidth=1, color="var(--primary-color)"),
        line=alt.LineConfig(strokeWidth=2, color="var(--primary-color)") # Use CSS variable if possible by theme
    ).encode(
        x=alt.X(f'{x_col}:T', title=x_title, axis=alt.Axis(format='%b %d', labelAngle=-45, grid=True)),
        y=alt.Y(f'{y_col}:Q', title=y_title, scale=alt.Scale(zero=False), axis=alt.Axis(grid=True)),
        tooltip=[
            alt.Tooltip(f'{x_col}:T', title=x_title, format='%Y-%m-%d'), 
            alt.Tooltip(f'{y_col}:Q', title=y_title, format='$.2f' if '$' in y_title else '.2f')
        ]
    ).properties(
        height=chart_height,
        # title="Market Trend" # Optional: add title to chart properties if needed
    ).interactive(bind_y=False) # Allow x-axis zoom/pan, fix y-axis
    
    st.altair_chart(chart, use_container_width=True, theme="streamlit")


def render_tabs(tabs_content: Dict[str, callable], default_tab_index: int = 0):
    """
    Renders Streamlit tabs dynamically from a dictionary.
    Key: Tab title, Value: Callable function to render tab content.
    """
    if not tabs_content:
        return

    tab_titles = list(tabs_content.keys())
    
    # Streamlit st.tabs doesn't have a direct default selection mechanism via index after creation.
    # The first tab is selected by default.
    # If a specific default is needed, it might require reordering `tab_titles`
    # or using query params to trigger a rerun and select a tab (more complex).
    
    selected_tabs = st.tabs(tab_titles) # Creates tabs
    
    for i, tab_title in enumerate(tab_titles):
        with selected_tabs[i]: # Context manager for each tab
            try:
                tabs_content[tab_title]() # Call the function to render content
            except Exception as e:
                st.error(f"Error rendering content for tab '{tab_title}': {e}")
                logger.error(f"Error in render_tabs for '{tab_title}': {e}", exc_info=True)