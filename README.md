# 🧠 TumorVision: Brain Tumor Detection & Clinical AI Intelligence

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Google Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-Multimodal_XAI-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)

**TumorVision** is a production-modernized clinical deep learning and Explainable AI (XAI) platform designed to classify brain tumors from axial MRI scans into four diagnostic categories:
1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **No Tumor (Healthy Control)**

In addition to categorical classification, TumorVision features visual **gradient-based Saliency Maps** and an automated **Gemini 1.5 Flash Neuroradiologist Explainer**, bridging the trust gap between black-box AI and clinical decision support.

---

## 🏗️ Project Architecture

```
tumor-vision/
├── backend/
│   ├── api.py                   # FastAPI REST API (/api/diagnose, /api/health, /api/samples)
│   ├── config.py                # Centralized paths, classes, visual tokens, thresholds
│   ├── models/                  # Storage directory for trained weights
│   │   └── README.md            # Weights download & placement instructions
│   └── services/
│       ├── inference.py         # Unified model loaders & inference engine
│       ├── saliency.py          # GradientTape & Saliency Heatmap generator
│       └── gemini_explainer.py  # Gemini 1.5 Flash multimodal clinical explainer
├── frontend/
│   ├── index.html               # Modern Glassmorphic Web App (Vanilla JS/HTML5)
│   ├── style.css                # Dark-mode clinical design system
│   └── app.js                   # Client interactions, sample chips, and charts
├── data/
│   └── sample_scans/            # Pre-packaged validation scans for instant testing
│       ├── glioma_sample.png
│       ├── meningioma_sample.png
│       ├── pituitary_sample.png
│       └── notumor_sample.png
├── app.py                       # Modernized, bug-fixed Streamlit application
├── run_app.py                   # System verification & one-click launcher
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Environment configuration template
├── .gitignore                   # Excludes .env, large weights, and cache
└── SIC_Capstone_Project (3).ipynb # Original Google Colab research notebook
```

---

## 🛠️ Issues Identified & Fixed During Modernization

| Issue | Severity | Status in Legacy Notebook | Fixed Implementation in Modernized Codebase |
| :--- | :---: | :--- | :--- |
| **Exposed Gemini API Key** | 🚨 Critical | Hardcoded plain-text API key in cell 127 (`app.py`) | Removed from source. Safely loaded from `.env` via `python-dotenv` with user-override option in UI. |
| **Colab Hardcoded Paths** | ⚠️ High | Fixed `/content/...` paths crashing local execution | Implemented portable `pathlib.Path` resolution across `backend/config.py`. |
| **Plotly Bar Chart Bug** | 🐛 High | `y=1` hardcoded in annotation loop, stacking all 4 labels | Replaced with `y=sorted_labels[i]`, color-coded to diagnostic classes. |
| **Missing Model Fallbacks** | ⚠️ Medium | Application crashes if `.h5` files are missing locally | Built smart **Diagnostic Simulation Mode** to test UI and Gemini explanations immediately. |
| **Monolithic Notebook** | ⚠️ Medium | Entire system locked in 136-cell notebook | Extracted into clean modular services (`inference`, `saliency`, `gemini_explainer`). |
| **Lack of Test Data** | ℹ️ Minor | Requires downloading 7GB Kaggle dataset | Added `data/sample_scans/` with pre-loaded validation samples for one-click testing. |

---

## 🔬 Deep Learning Benchmark

| Metric | Transfer Learning (Xception) | Custom Sequential CNN |
| :--- | :---: | :---: |
| **Input Shape** | **299 × 299 × 3** | **224 × 224 × 3** |
| **Pre-trained Weights** | ImageNet | Trained from scratch |
| **Training Accuracy** | **99.84%** | 98.86% |
| **Validation Accuracy** | **98.28%** | 93.61% |
| **Test Accuracy** | **100.00%** | 96.20% |
| **Optimizer & Loss** | Adamax (0.001), Categorical Cross-Entropy | Adamax (0.001), Categorical Cross-Entropy |

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the repository, create a virtual environment, and install dependencies:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🔄 Understanding Modes: Diagnostic Simulation vs. 100% Live AI

TumorVision includes an automatic dual-engine design to ensure the application works out-of-the-box while allowing a seamless transition to live deep learning:

| Feature | 🔵 Simulation / Validation Mode (Default) | 🟢 100% Live Deep Learning Mode |
| :--- | :--- | :--- |
| **When Activated** | When `backend/models/` is empty or `.env` key is blank | When model files are present & API key is set |
| **Neural Inference** | Anatomical luminance & regional density heuristics | Live TensorFlow neural backpropagation on MRI tensors |
| **Saliency Maps** | Sobel intracranial gradient projection | Exact `tf.GradientTape` neuron attribution gradients |
| **Neurological Commentary**| Evidence-based offline neurological knowledge base | Real-time multimodal Google Gemini 1.5 Flash agent |
| **UI Badge** | `● Simulation / Validation Mode` (Blue) | `● Live Deep Learning` (Green) |

---

## 🛠️ Step-by-Step Guide: Switch to 100% Real-Time Live Mode

Follow these 2 simple steps to activate live deep learning and real-time Gemini AI:

### Step 1: Add Your Trained Deep Learning Weights
From your Google Colab training notebook, download your exported weight files:
1. **Xception Model**: Download `xception_model.weights.h5` (Cell 109: `model.save_weights("xception_model.weights.h5")`).
2. **Custom CNN**: Download `cnn_model.h5` (Cell 99: `cnn_model.save("cnn_model.h5")`).
3. Move or copy these files into the [`backend/models/`](backend/models/) folder:
   ```
   tumor-vision/backend/models/
   ├── xception_model.weights.h5   # (Place here)
   └── cnn_model.h5                # (Place here)
   ```
*The inference engine will automatically detect these files on launch, load the TensorFlow architecture, and switch classification to live weights.*

### Step 2: Connect Live Google Gemini 1.5 Flash
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Open the [`.env`](.env) file in the root directory:
   ```env
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   ```
3. Save the file.
*(Alternatively, you can paste the key directly into the **"Gemini AI Key"** button in the web navbar or sidebar).*

### Step 3: Verify Live Activation
Run the launcher:
```bash
python run_app.py --mode api
```
The terminal check will output:
```
✓ Xception Transfer Learning weights found.
✓ Live Gemini API Key detected.
```
And the web dashboard at `http://127.0.0.1:8000/` will now display the **`● Live Deep Learning`** badge!

### 4. Launch Applications

#### A. Modern Streamlit UI (Recommended)
```bash
python run_app.py --mode streamlit
```
*Opens interactive interface at `http://localhost:8501`.*

#### B. FastAPI REST Service & Modern Web Portal
```bash
python run_app.py --mode api
```
*Starts API server at `http://127.0.0.1:8000` (Interactive docs at `/docs`).*  
*To view the Modern Web Console, open [`frontend/index.html`](frontend/index.html) in your browser.*

---

## 📡 REST API Reference

### Health Check
`GET /api/health`
```json
{
  "status": "healthy",
  "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
  "weights": { "xception_weights_found": true, "cnn_model_found": true }
}
```

### Run Comprehensive Diagnosis
`POST /api/diagnose` (Multipart Form: `file`, `model_choice`, `api_key`)
```json
{
  "prediction": {
    "class": "Pituitary",
    "confidence": 0.9942,
    "probabilities": { "Glioma": 0.0031, "Meningioma": 0.0022, "No Tumor": 0.0005, "Pituitary": 0.9942 }
  },
  "saliency_map": {
    "image_base64": "data:image/png;base64,..."
  },
  "neurologist_assessment": "Focal gradient hyperdensities concentrate within the sellar fossa..."
}
```

---

## ⚖️ Clinical Disclaimer
TumorVision is designed for research, academic evaluation, and clinical decision support. All AI-generated predictions and explanations must be verified by a licensed neuroradiologist or medical practitioner.
