import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_llm():
    return pipeline(
        task="text-generation",
        model="google/flan-t5-small",
        return_full_text=False   # ⭐ THIS IS THE KEY
    )
