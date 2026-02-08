import streamlit as st
import os
import importlib
from PIL import Image
from pathlib import Path
import helper  # Assuming your helper.py handles YOLO loading
import settings
import inspect
from detection_client import get_groq_client, extract_content

# --- INITIALIZATION ---
st.set_page_config(page_title="AgriVision AI", page_icon="🌿", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Upload & Confidence ---
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence (%)", 25, 100, 45) / 100.0
source_img = st.sidebar.file_uploader("Upload Leaf", type=("jpg", "jpeg", "png"))

# Load Model
@st.cache_resource
def load_yolo():
    return helper.load_model(Path(settings.DETECTION_MODEL))


def show_version_warnings():
    issues = []
    # streamlit version
    try:
        import streamlit as _st
        st_ver = getattr(_st, "__version__", "?")
    except Exception:
        st_ver = "(not importable)"
    # numpy
    try:
        import numpy as _np
        np_ver = _np.__version__
    except Exception:
        np_ver = "(not importable)"
    # opencv
    try:
        import cv2 as _cv
        cv_ver = _cv.__version__
    except Exception:
        cv_ver = None

    if cv_ver and np_ver and np_ver[0] != '2':
        # opencv wheels often require numpy >=2; warn if mismatch
        try:
            major = int(np_ver.split('.')[0])
            if major < 2:
                issues.append(f"OpenCV {cv_ver} may require numpy>=2 but numpy is {np_ver}.")
        except Exception:
            pass

    if issues:
        for it in issues:
            st.warning(it)


# Do not load the heavy model at import-time; load lazily inside the handler per DESIGN.md

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Image Analysis")
    # show potential version/compatibility warnings for runtime deps
    try:
        show_version_warnings()
    except Exception:
        pass
    if source_img:
        uploaded_image = Image.open(source_img)
        # compatibility wrapper: some streamlit versions accept use_container_width, others use use_column_width
        def st_image_compat(*args, **kwargs):
            try:
                sig = inspect.signature(st.image)
                if 'use_container_width' in sig.parameters:
                    # ensure the arg is provided; prefer use_container_width
                    if 'use_container_width' not in kwargs and 'use_column_width' in kwargs:
                        kwargs['use_container_width'] = kwargs.pop('use_column_width')
                    elif 'use_container_width' not in kwargs:
                        kwargs['use_container_width'] = True
                else:
                    # fallback to older name
                    if 'use_column_width' not in kwargs and 'use_container_width' in kwargs:
                        kwargs['use_column_width'] = kwargs.pop('use_container_width')
                return st.image(*args, **kwargs)
            except Exception:
                # last-resort call without width args
                kwargs.pop('use_container_width', None)
                kwargs.pop('use_column_width', None)
                return st.image(*args, **kwargs)

        # Show uploaded image (use explicit width to avoid deprecated args)
        st.image(uploaded_image, caption="Uploaded", width=420)

        if st.button("Detect Disease"):
            # Lazy-load the model at inference time
            yolo_model = load_yolo()
            results = None
            try:
                results = yolo_model.predict(uploaded_image, conf=confidence)
            except Exception as e:
                st.error(f"Model inference failed: {e}")

            if results:
                try:
                    plotted_img = results[0].plot()
                    # plotted_img may be BGR numpy
                    try:
                        import numpy as _np
                        if _np.array(plotted_img).ndim == 3:
                            plotted_img = plotted_img[:, :, ::-1]
                    except Exception:
                        pass
                    st.image(plotted_img, caption="Result", width=700)
                except Exception:
                    st.warning("Could not plot model output visually; showing textual detections below.")

                # Extract results for the LLM
                detections = []
                first = results[0]
                for box in getattr(first, 'boxes', []):
                    # robust class extraction
                    try:
                        cval = getattr(box, 'cls')
                        if hasattr(cval, '__len__'):
                            cidx = int(cval[0])
                        else:
                            cidx = int(cval)
                    except Exception:
                        cidx = None
                    try:
                        if hasattr(first, 'names'):
                            names_map = first.names
                            try:
                                cls_name = names_map[cidx]
                            except Exception:
                                cls_name = str(cidx)
                        else:
                            cls_name = str(cidx)
                    except Exception:
                        cls_name = str(cidx)
                    try:
                        confv = getattr(box, 'conf')
                        if hasattr(confv, '__len__'):
                            conff = float(confv[0])
                        else:
                            conff = float(confv)
                    except Exception:
                        conff = 0.0
                    detections.append(f"{cls_name} ({conff*100:.2f}%)")

                if detections:
                    detection_text = ", ".join(detections)
                    # Create the custom prompt for the AI
                    ai_prompt = (
                        f"The system detected: {detection_text}. \nTasks:\n"
                        "1. Explain the disease in farmer-friendly language.\n"
                        "2. Mention causes.\n3. Suggest immediate actions.\n"
                        "4. If confidence is below 60%, mention uncertainty."
                    )
                    # Push this to session state so the chat picks it up
                    st.session_state.messages.append({"role": "user", "content": ai_prompt})
                else:
                    st.warning("No disease detected.")

with col2:
    st.subheader("💬 AI Plant Doctor")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input (if user wants to ask follow-up questions)
    if chat_input := st.chat_input("Ask more about the treatment..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

    # Generate response if the last message is from 'user'
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Use centralized get_groq_client (detection_client.py) per DESIGN.md
            client = get_groq_client()
            if client is None:
                st.warning("No GROQ API key found — the AI assistant is disabled. Add GROQ_API_KEY to Streamlit secrets or environment variables to enable.")
                # Provide a simple fallback reply to avoid blocking the UI
                fallback = "I don't have access to the external AI right now. Please add your API key or try again later."
                response_placeholder.markdown(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})
            else:
                try:
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=st.session_state.messages,
                        stream=True,
                    )
                    for chunk in completion:
                        c = extract_content(chunk)
                        if c:
                            full_response += c
                            response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"AI completion failed: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})
