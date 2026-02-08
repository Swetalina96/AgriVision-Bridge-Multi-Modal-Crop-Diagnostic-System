import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AgriVision-Bridge",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# TITLE & INTRO
# --------------------------------------------------
st.title("🌾 AgriVision-Bridge")
st.subheader("Multi-Modal Crop Diagnostic System")

st.markdown(
    """
AgriVision-Bridge is an end-to-end **AI-powered crop disease diagnostic system**
that combines **computer vision** and **transformer-based language models**
to deliver **explainable, farmer-friendly insights** from crop images.
"""
)

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------
st.markdown("## 📌 Project Overview")

st.markdown(
    """
Traditional crop disease diagnosis is time-consuming and often requires
expert knowledge. AgriVision-Bridge addresses this challenge by using:

- **YOLO-based object detection** to identify crop diseases from images
- **Local Large Language Models (LLMs)** to generate human-readable diagnostics
- A **fully offline, deployable pipeline** suitable for real-world agri-advisory use
"""
)

# --------------------------------------------------
# SYSTEM ARCHITECTURE
# --------------------------------------------------
st.markdown("## 🧠 System Architecture")

st.markdown(
    """
### 🔍 Vision Layer
- Detects crop diseases using a **custom-trained YOLO model**
- Outputs disease labels and confidence scores

### 🧠 Reasoning Layer
- Converts YOLO outputs into structured prompts
- Uses a **local transformer-based LLM (Gemma 2B)** for explainable reasoning
- Handles low-confidence detections responsibly

### 🔗 Integration Layer
- Streamlit-based multi-page interface
- Visual detection + textual diagnosis
- Fully local and production-ready
"""
)

# --------------------------------------------------
# BUSINESS USE CASES
# --------------------------------------------------
st.markdown("## 🚜 Business Use Cases")

st.markdown(
    """
- AI-assisted crop health diagnostics for farmers  
- Early disease detection and yield protection  
- Agri-advisory platforms and decision support systems  
- Smart farming and precision agriculture  
"""
)

# --------------------------------------------------
# SKILLS & TECH STACK
# --------------------------------------------------
st.markdown("## 🛠️ Skills & Technologies")

st.markdown(
    """
**Computer Vision:** YOLOv8, Image Processing  
**Generative AI:** Transformers, Prompt Engineering  
**Multi-Modal AI:** Vision + Language Integration  
**Deployment:** Streamlit, Local Inference  
**Acceleration:** GPU/CPU benchmarking  
"""
)

# --------------------------------------------------
# NAVIGATION HELP
# --------------------------------------------------
st.markdown("## 👉 How to Use This App")

st.info(
    """
Use the **sidebar navigation** to explore:
- **Object Detection:** Upload images and receive AI-powered diagnosis
- **Model Performance:** View training metrics, confusion matrix, and results
"""
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© 2026 | AgriVision-Bridge | Multi-Modal AI for Smart Agriculture 🌱")
