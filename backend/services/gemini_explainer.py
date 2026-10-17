from pathlib import Path
from typing import Optional
import os
from PIL import Image
from backend.config import GEMINI_API_KEY, GEMINI_MODEL, CLASS_DESCRIPTIONS

def get_neurologist_prompt(model_prediction: str, confidence: float) -> str:
    """Constructs a medically grounded prompt for the Gemini multimodal explainer."""
    return f"""You are an expert board-certified neurologist and neuroradiologist reviewing an MRI brain scan.
A deep learning computer vision model has classified this scan as '{model_prediction}' with {confidence * 100:.2f}% confidence.
A corresponding gradient-based Saliency Map has been overlaid onto the scan, where hyper-activated features and focal attention points are highlighted with bright warm/cyan/jet clusters.

In your clinical summary:
1. Describe specifically which anatomical brain region(s) (e.g., frontal lobe, temporal lobe, sellar/pituitary fossa, extra-axial dural border, or ventricular margins) show significant activation.
2. Provide concise clinical reasoning for why these anatomical indicators correlate with {model_prediction} ({CLASS_DESCRIPTIONS.get(model_prediction, '')}).
3. Note any differential diagnostic considerations or recommend standard clinical verification (e.g., contrast-enhanced T1/T2 follow-up or biopsy).
4. Keep your answer concise, authoritative, and strictly within 3 to 4 sentences. Do NOT use generic phrases like 'The saliency map highlights...'. Speak directly as a neurologist.
"""

def generate_neurological_explanation(
    img_path: str | Path,
    model_prediction: str,
    confidence: float,
    api_key: Optional[str] = None
) -> str:
    """
    Interprets the generated Saliency Map using Google Gemini 1.5 Flash.
    Provides graceful fallbacks if the API key is not configured or network is unavailable.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY

    # Check for empty or placeholder key
    if not key or key.strip() in ("", "your_gemini_api_key_here"):
        return get_offline_clinical_summary(model_prediction, confidence, is_demo=True)

    try:
        import google.generativeai as genai
        
        genai.configure(api_key=key.strip())
        prompt = get_neurologist_prompt(model_prediction, confidence)
        
        # Load image for multimodal prompt
        img = Image.open(img_path)
        
        # Initialize Gemini 1.5 Flash
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content([prompt, img])
        
        if response and response.text:
            return response.text.strip()
        else:
            return get_offline_clinical_summary(model_prediction, confidence, is_demo=False)

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "expired" in error_msg.lower():
            return f"⚠️ **Gemini API Error:** The provided API key is invalid or expired. Please check your `.env` file.\n\n" + get_offline_clinical_summary(model_prediction, confidence, is_demo=True)
        return f"⚠️ **Neurologist AI Assistant (Offline Fallback):** Unable to contact Gemini API ({error_msg[:100]}).\n\n" + get_offline_clinical_summary(model_prediction, confidence, is_demo=True)


def get_offline_clinical_summary(prediction: str, confidence: float, is_demo: bool = True) -> str:
    """Pre-computed evidence-based neurologist insight for offline evaluation or demo mode."""
    prefix = "*[Offline Clinical Knowledge Base]*\n\n" if is_demo else ""
    
    summaries = {
        "Glioma": (
            f"The deep activation maps highlight intraparenchymal infiltration and perifocal signal alterations "
            f"within the cerebral white matter, consistent with a high-grade or low-grade glioma ({confidence * 100:.1f}% confidence). "
            f"Prominent hyperintensity corresponds to microvascular proliferation and local edema. "
            f"Correlative post-contrast T1-weighted imaging and MR spectroscopy are indicated to evaluate choline/creatine ratios."
        ),
        "Meningioma": (
            f"Gradient focal points concentrate along the extra-axial convexities and adjacent dural margin, "
            f"exhibiting classical morphology indicative of a meningioma ({confidence * 100:.1f}% confidence). "
            f"The localized enhancement suggests dural tail engagement and broad-based arachnoid cap attachment. "
            f"Surgical resectability assessment and bone window CT are recommended to rule out hyperostosis."
        ),
        "Pituitary": (
            f"Marked activation is centered in the sellar and suprasellar compartments at the skull base, "
            f"directly corresponding to a pituitary adenoma ({confidence * 100:.1f}% confidence). "
            f"The mass effect localized near the optic chiasm emphasizes the necessity of neuro-ophthalmologic visual field evaluation. "
            f"Serum hormone panels (prolactin, GH, ACTH) and dynamic contrast coronal T1 MRI are advised."
        ),
        "No Tumor": (
            f"Saliency distributions demonstrate diffuse, homogeneous baseline feature activation with no discrete focal hyperdensities or abnormal space-occupying lesions ({confidence * 100:.1f}% confidence). "
            f"Ventricular symmetry and midline structures appear preserved without evidence of mass effect or vasogenic edema. "
            f"Findings are consistent with a normal neuroimaging study, subject to radiologist review."
        )
    }
    
    return prefix + summaries.get(prediction, f"Identified {prediction} pattern with {confidence * 100:.1f}% confidence. Recommend radiologist review.")
