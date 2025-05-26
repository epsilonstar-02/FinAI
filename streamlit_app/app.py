import streamlit as st

st.set_page_config(page_title="Morning Market Brief")
st.title("🌅 Morning Market Brief")

st.sidebar.header("Input")
user_input = st.sidebar.text_input("Ask your brief:", "What’s our risk exposure today?")
if st.sidebar.button("Run"):
    st.write("…loading…")  # placeholder
