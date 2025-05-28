import time
import streamlit as st


def render_card(title, content):
    """
    Renders a styled card with title and content.
    
    Args:
        title: The card title
        content: The card content text
    """
    card_html = f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_audio_player(audio_bytes):
    """
    Renders an audio player with the provided audio bytes.
    
    Args:
        audio_bytes: The audio data in bytes
    """
    st.audio(audio_bytes, format="audio/mp3")


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
    st.success(f"{name} completed in {elapsed:.2f}s")
