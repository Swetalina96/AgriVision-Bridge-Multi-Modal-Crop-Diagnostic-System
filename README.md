# 🌾 AgriVision-Bridge: Multi-Modal Crop Diagnostic System

AgriVision-Bridge is an end-to-end **AI-powered crop disease diagnostic system** that integrates **computer vision (YOLO)** with **large language models (LLMs)** to deliver **explainable, farmer-friendly insights** from crop leaf images.

The system is designed to work as a **fully deployable, production-ready multi-modal AI application**, suitable for real-world agri-advisory and precision agriculture use cases.

---

## 📌 Project Overview

Traditional crop disease diagnosis is time-consuming, expert-dependent, and often inaccessible to small farmers. AgriVision-Bridge addresses this gap by combining:

* **YOLO-based object detection** to identify crop diseases from images
* **Transformer-based LLM reasoning** to generate understandable diagnostic reports
* A **Streamlit-powered interface** for seamless user interaction

The result is a **vision + language AI pipeline** that not only detects diseases but also explains *what*, *why*, and *what to do next*.

---

## 🧠 System Architecture

### 🔍 Vision Layer

* Custom-trained **YOLO model** for crop disease detection
* Outputs disease class labels with confidence scores
* GPU-accelerated inference with CPU/GPU benchmarking

### 🧠 Reasoning Layer

* Converts YOLO outputs into structured prompts
* Uses a **local or API-based LLM (LLaMA 3 / Mistral / Gemma)**
* Handles low-confidence detections using prompt logic
* Generates farmer-friendly explanations, causes, and action plans

### 🔗 Integration Layer

* **Streamlit multi-page application**
* Visual bounding boxes + conversational AI diagnosis
* Modular, clean, and deployment-ready design

---

## 🚜 Business Use Cases

* AI-assisted crop health diagnostics for farmers
* Early disease detection and yield protection
* Agri-advisory and decision support platforms
* Smart farming and precision agriculture solutions

---

## 🛠️ Skills & Technologies

**Computer Vision**

* YOLO (object detection)
* Image preprocessing & augmentation

**Generative AI**

* Transformers
* Prompt engineering
* Local LLM deployment

**Multi-Modal AI**

* Vision + Language pipelines

**Deployment & Tools**

* Streamlit
* GPU / CPU benchmarking
* Modular Python architecture

---

## 📂 Project Structure

```text
AgriVision-Bridge-Multi-Modal-Crop-Diagnostic-System/
│
├── app.py                     # Main Streamlit entry point
├── helper.py                  # YOLO loading & inference helpers
├── settings.py                # Model & path configurations
├── results.csv                # YOLO training metrics
├── results.png                # Training curves
├── confusion_matrix.png       # Confusion matrix
│
├── pages/
│   ├── object_detection.py    # Image upload + AI diagnosis page
│   └── model_performance.py   # Model evaluation & benchmarking
│
└── README.md
```

---

## 📊 Model Performance & Evaluation

The YOLO vision model is evaluated using:

* **mAP@50 and mAP@50–95**
* **Precision & Recall**
* **Training and validation loss curves**
* **Confusion matrix analysis**

These metrics validate the robustness of the vision layer before passing detections to the LLM reasoning module.

---

## 📈 Key Results

* Accurate crop disease detection from leaf images
* Explainable AI-generated diagnostic reports
* GPU-optimized inference pipeline
* Fully deployable, end-to-end multi-modal AI system

---

## 📊 Project Evaluation Metrics

* Detection accuracy (mAP, confidence quality)
* Inference latency (CPU vs GPU)
* Prompt robustness for low-confidence detections
* End-to-end system integration quality

---

## 🌱 Dataset

* **Source:** Public crop disease image datasets (PlantVillage)
* **Size:** 10,000+ images
* **Format:** Images + annotations
* **Classes:** Multiple crop diseases

**Dataset Link:**
[https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### Dataset Preprocessing

* Image resizing & normalization
* Data augmentation
* Label verification
* Train / validation split
* Annotation consistency checks

---

## ▶️ How to Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar navigation to:

* Upload crop images and detect diseases
* View AI-generated diagnostic explanations
* Analyze model performance and benchmarks

---

## 📦 Project Deliverables

* Python source code
* Trained YOLO model
* Streamlit application
* Model evaluation artifacts
* This README documentation

---

## ⏱️ Timeline

* **Duration:** 2 weeks

---

## 📌 Domain

AgriTech | Computer Vision | Generative AI | Multi-Modal AI

---

## 📜 License & Usage

This project is intended for **educational and research purposes**. Dataset usage follows respective dataset licenses.

---

© 2026 | **AgriVision-Bridge** | Multi-Modal AI for Smart Agriculture 🌱
