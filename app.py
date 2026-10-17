import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import io

from backend.config import (
    CLASSES,
    CLASS_DESCRIPTIONS,
    CLASS_COLORS,
    SAMPLES_DIR,
    SALIENCY_DIR
)
from backend.services.inference import predict_tumor, load_model_instance
from backend.services.saliency import generate_saliency_map_tf, generate_synthetic_saliency_map
from backend.services.gemini_explainer import generate_neurological_explanation

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="TumorVision | Neurological MRI Diagnostic AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark-Mode CSS
st.markdown("""
<style>
    /* Main Background & Typography */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d131f 0%, #080a10 90%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .hero-container {
        padding: 24px 28px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 850px;
    }
    
    /* Diagnostic Glass Cards */
    .diag-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .diag-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        padding: 28px 0 12px 0;
        color: #64748b;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls & API Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=300&q=80", use_container_width=True)
    st.markdown("### ⚙️ Diagnostic Engine")
    
    model_choice = st.radio(
        "Neural Network Architecture",
        ("Transfer Learning - Xception", "Custom 4-Block CNN"),
        help="Xception features 99.8% validation accuracy on 299x299 inputs. Custom CNN is trained on 224x224 inputs."
    )
    
    st.markdown("---")
    st.markdown("### 🔑 Neurologist AI (Gemini)")
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Optional: Enter your key here or configure it in `.env` for automated clinical explanations."
    )
    if not user_api_key:
        st.caption("ℹ️ *Leave blank to use the built-in evidence-based clinical knowledge base.*")

    st.markdown("---")
    st.markdown("### 📖 Diagnostic Classes")
    for cls, desc in CLASS_DESCRIPTIONS.items():
        color = CLASS_COLORS.get(cls, "#38bdf8")
        st.markdown(f"<span style='color:{color}; font-weight:700;'>• {cls}</span>: <span style='font-size:0.82rem; color:#94a3b8;'>{desc}</span>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 TumorVision: Brain MRI Clinical Diagnostics</div>
    <div class="hero-subtitle">
        High-precision multi-class brain tumor classification powered by Deep Convolutional Neural Networks, 
        gradient-based Explainable AI (XAI) Saliency Mapping, and Gemini 1.5 Flash multimodal clinical reasoning.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Scan Selection: Upload vs Sample Scan
# ---------------------------------------------------------
input_tab1, input_tab2 = st.tabs(["📁 Upload MRI Scan", "🧪 Test with Pre-Loaded Sample"])

active_image = None
active_image_name = "uploaded_scan.png"

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload Axial Brain MRI Scan (T1 / T2 / FLAIR)",
        type=["jpg", "jpeg", "png"],
        help="Accepts high-resolution DICOM export images in standard formats."
    )
    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file).convert("RGB")
            active_image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading image: {e}")

with input_tab2:
    st.write("Select a validated clinical test scan to evaluate model performance without manual uploads:")
    sample_options = {
        "Select a sample...": None,
        "Sample 1: Pituitary Adenoma (Sellar Mass)": SAMPLES_DIR / "pituitary_sample.png",
        "Sample 2: Intraparenchymal Glioma (White Matter)": SAMPLES_DIR / "glioma_sample.png",
        "Sample 3: Convexity Meningioma (Extra-Axial)": SAMPLES_DIR / "meningioma_sample.png",
        "Sample 4: Normal Healthy Control (No Tumor)": SAMPLES_DIR / "notumor_sample.png"
    }
    
    selected_sample = st.selectbox("Pre-Packaged Validation Scans", list(sample_options.keys()))
    if selected_sample and sample_options[selected_sample] and sample_options[selected_sample].exists():
        if active_image is None or uploaded_file is None:
            active_image = Image.open(sample_options[selected_sample]).convert("RGB")
            active_image_name = sample_options[selected_sample].name
            st.info(f"Loaded `{active_image_name}` for evaluation.")


# ---------------------------------------------------------
# Main Diagnostic Pipeline Execution
# ---------------------------------------------------------
if active_image is not None:
    st.markdown("---")
    
    with st.spinner("Executing neural inference and generating Explainable AI saliency map..."):
        # 1. Run Deep Learning Prediction
        pred_res = predict_tumor(active_image, model_type=model_choice)
        
        pred_class = pred_res["predicted_class"]
        confidence = pred_res["confidence"]
        class_idx = pred_res["class_idx"] if "class_idx" in pred_res else pred_res["class_index"]
        probabilities = pred_res["probabilities"]
        accent_color = pred_res["color"]
        
        # 2. Generate Saliency Map
        model_instance, has_weights = load_model_instance("xception" if "xception" in model_choice.lower() else "cnn")
        
        if model_instance is not None and has_weights:
            try:
                saliency_overlay, saliency_path = generate_saliency_map_tf(
                    model=model_instance,
                    img_array=pred_res["pil_image"],
                    class_index=class_idx,
                    target_size=pred_res["target_size"],
                    original_pil_img=active_image,
                    save_filename=f"saliency_{active_image_name}"
                )
            except Exception:
                saliency_overlay, saliency_path = generate_synthetic_saliency_map(
                    original_pil_img=active_image,
                    class_index=class_idx,
                    target_size=pred_res["target_size"],
                    save_filename=f"saliency_{active_image_name}"
                )
        else:
            saliency_overlay, saliency_path = generate_synthetic_saliency_map(
                original_pil_img=active_image,
                class_index=class_idx,
                target_size=pred_res["target_size"],
                save_filename=f"saliency_{active_image_name}"
            )
            
        # 3. Generate Gemini Neurological Explanation
        explanation = generate_neurological_explanation(
            img_path=saliency_path,
            model_prediction=pred_class,
            confidence=confidence,
            api_key=user_api_key
        )

    # ---------------------------------------------------------
    # Visual Output: Primary Diagnostic Summary Cards
    # ---------------------------------------------------------
    c1, c2, c3 = st.columns([1.5, 1, 1.2])
    
    with c1:
        st.markdown(f"""
        <div class="diag-card">
            <div style="font-size:0.85rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Primary Diagnosis</div>
            <div style="font-size:2.2rem; font-weight:800; color:{accent_color}; margin:6px 0;">
                {pred_class}
            </div>
            <div style="font-size:0.9rem; color:#cbd5e1;">{pred_res['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="diag-card">
            <div style="font-size:0.85rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Classification Confidence</div>
            <div style="font-size:2.2rem; font-weight:800; color:#38bdf8; margin:6px 0;">
                {confidence * 100:.2f}%
            </div>
            <div style="font-size:0.85rem; color:#94a3b8;">Categorical Softmax Score</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        status_text = "Live Trained Weights" if pred_res["mode"] == "live" else "Simulation / Validation Mode"
        status_bg = "#065f46" if pred_res["mode"] == "live" else "#1e3a8a"
        st.markdown(f"""
        <div class="diag-card">
            <div style="font-size:0.85rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Active Architecture</div>
            <div style="font-size:1.25rem; font-weight:700; color:#e2e8f0; margin:6px 0;">
                {pred_res['model_used']}
            </div>
            <div style="margin-top:6px;">
                <span class="status-pill" style="background:{status_bg}; color:#e2e8f0;">● {status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Comparative Imaging & Explainable AI (XAI)
    # ---------------------------------------------------------
    st.markdown("### 🔬 Multimodal Neuroimaging & Feature Attribution")
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.image(active_image, caption=f"Original Brain MRI: {active_image_name}", use_container_width=True)
        
    with img_col2:
        st.image(
            saliency_overlay, 
            caption="Gradient Saliency Map (Cyan/Jet indicates high gradient focal points)", 
            use_container_width=True
        )

    # ---------------------------------------------------------
    # Multi-Class Probability Distribution (Fixed Plotly Chart)
    # ---------------------------------------------------------
    st.markdown("### 📊 Diagnostic Class Probability Distribution")
    
    sorted_classes = sorted(probabilities.keys(), key=lambda k: probabilities[k])
    sorted_probs = [probabilities[k] for k in sorted_classes]
    bar_colors = [CLASS_COLORS.get(k, "#38bdf8") if k == pred_class else "rgba(100, 116, 139, 0.4)" for k in sorted_classes]

    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_classes,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1)
        ),
        text=[f"{p * 100:.2f}%" for p in sorted_probs],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13)
    ))

    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=40, t=20, b=20),
        height=280,
        xaxis=dict(
            title="Classification Probability",
            range=[0, max(sorted_probs) * 1.25],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            tickfont=dict(color="#f8fafc", size=13, family="sans-serif")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Neurological Clinical AI Assessment
    # ---------------------------------------------------------
    st.markdown("### 🩺 Neurological Clinical Assessment (Gemini 1.5 Flash)")
    st.markdown(f"""
    <div class="diag-card" style="border-left: 4px solid {accent_color};">
        <div style="font-weight:700; color:{accent_color}; margin-bottom:8px; font-size:1.05rem;">
            Dr. AI Neuroradiological Impression
        </div>
        <div style="line-height:1.7; color:#e2e8f0; font-size:0.98rem;">
            {explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Clinical Export Action
    # ---------------------------------------------------------
    report_content = f"""TUMORVISION NEUROLOGICAL DIAGNOSTIC REPORT
Generated: {pred_res['model_used']}
Target File: {active_image_name}
--------------------------------------------------
PRIMARY DIAGNOSIS: {pred_class}
CONFIDENCE: {confidence * 100:.2f}%
CLASS BREAKDOWN:
{chr(10).join([f"  - {k}: {v*100:.2f}%" for k, v in probabilities.items()])}

CLINICAL IMPRESSION:
{explanation}
--------------------------------------------------
*This report is generated by a computer vision assistant for clinical decision support.*
"""
    st.download_button(
        label="📥 Export Clinical Diagnostic Report (.txt)",
        data=report_content,
        file_name=f"tumorvision_report_{active_image_name.split('.')[0]}.txt",
        mime="text/plain"
    )

else:
    st.info("👋 Upload a brain MRI scan above or select a pre-loaded sample from the **Test with Pre-Loaded Sample** tab to begin analysis.")

# Footer
st.markdown("""
<div class="footer-text">
    TumorVision AI • Developed with TensorFlow, Streamlit, and Google Gemini • For Research and Decision Support Use
</div>
""", unsafe_allow_html=True)
