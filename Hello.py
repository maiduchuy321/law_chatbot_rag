import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# RAG-based Chatbot for Public Administration 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    **👈 Select a demo from the sidebar** to see some examples
"""
)

