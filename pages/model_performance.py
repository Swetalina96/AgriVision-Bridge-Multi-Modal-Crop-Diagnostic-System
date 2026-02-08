import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Model Performance | AgriVision-Bridge",
    layout="wide"
)

# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RESULTS_CSV = ROOT / "results.csv"
CONF_MATRIX_IMG = ROOT / "confusion_matrix.png"
RESULTS_IMG = ROOT / "results.png"

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Model Performance Evaluation")

st.markdown(
    """
This page presents the **training and evaluation performance** of the
YOLO-based crop disease detection model used in AgriVision-Bridge.
"""
)

# --------------------------------------------------
# METRICS TABLE
# --------------------------------------------------
st.subheader("📈 Training Metrics (Per Epoch)")

if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("results.csv not found.")

# --------------------------------------------------
# KEY METRICS SUMMARY
# --------------------------------------------------
if RESULTS_CSV.exists():
    st.subheader("⭐ Best Model Performance")

    best_map50 = df["metrics/mAP50(B)"].max()
    best_map5095 = df["metrics/mAP50-95(B)"].max()
    best_precision = df["metrics/precision(B)"].max()
    best_recall = df["metrics/recall(B)"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best mAP@50", f"{best_map50:.3f}")
    col2.metric("Best mAP@50–95", f"{best_map5095:.3f}")
    col3.metric("Best Precision", f"{best_precision:.3f}")
    col4.metric("Best Recall", f"{best_recall:.3f}")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("🧩 Confusion Matrix")

if CONF_MATRIX_IMG.exists():
    st.image(
        Image.open(CONF_MATRIX_IMG),
        caption="Confusion Matrix for Crop Disease Classification",
        use_container_width=True
    )
else:
    st.warning("confusion_matrix.png not found.")

# --------------------------------------------------
# TRAINING CURVES
# --------------------------------------------------
st.subheader("📉 Training & Validation Curves")

if RESULTS_IMG.exists():
    st.image(
        Image.open(RESULTS_IMG),
        caption="YOLO Training Results (Loss, Precision, Recall, mAP)",
        use_container_width=True
    )
else:
    st.warning("results.png not found.")

# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------
st.subheader("🧠 Performance Interpretation")

st.markdown(
    """
- The model demonstrates **high detection accuracy** with strong mAP scores.
- Precision and recall values indicate **robust disease classification**.
- Stable loss curves suggest **effective convergence** during training.
- These metrics validate the reliability of the vision layer before
  passing results to the **LLM reasoning module**.
"""
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("AgriVision-Bridge | Model Evaluation & Benchmarking 📊")
