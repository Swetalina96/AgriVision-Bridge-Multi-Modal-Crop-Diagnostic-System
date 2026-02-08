import os
from typing import Any

try:
    import streamlit as st
except Exception:
    st = None


def extract_content(chunk: Any) -> str | None:
    """Robustly extract streaming chunk text for dict- or object-style chunks.

    Supports dict chunks like {'choices':[{'delta':{'content': '...'}}]}
    and object chunks with attributes (chunk.choices[0].delta.content).
    Returns the content string or None if not present.
    """
    try:
        if isinstance(chunk, dict):
            choices = chunk.get('choices') or []
            if not choices:
                return None
            return choices[0].get('delta', {}).get('content')
        else:
            choices = getattr(chunk, 'choices', None) or []
            if not choices:
                return None
            delta = getattr(choices[0], 'delta', None)
            return getattr(delta, 'content', None)
    except Exception:
        return None


def get_groq_client():
    """Return a Groq client if GROQ_API_KEY is available, else None.

    Prefers Streamlit secrets then environment variables. Lazy-imports the SDK.
    Returns None if no key or client can't be constructed.
    """
    api_key = None
    try:
        if st is not None:
            sec = getattr(st, 'secrets', None)
            if sec:
                api_key = sec.get('GROQ_API_KEY') if hasattr(sec, 'get') else sec['GROQ_API_KEY'] if 'GROQ_API_KEY' in sec else None
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.environ.get('GROQ_API_KEY')

    if not api_key:
        return None

    try:
        from groq import Groq

        return Groq(api_key=api_key)
    except Exception:
        return None
