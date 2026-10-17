from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

BACKEND_DIR = ROOT_DIR / "backend"
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "sample_scans"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SALIENCY_DIR = ROOT_DIR / "saliency_maps"
SALIENCY_DIR.mkdir(parents=True, exist_ok=True)

# Diagnostic Class Labels
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Class Descriptions & Clinical Context
CLASS_DESCRIPTIONS = {
    "Glioma": "Tumor originating from glial cells within the brain tissue (astrocytoma, oligodendroglioma, or glioblastoma).",
    "Meningioma": "Typically benign, slow-growing tumor arising from the meningeal coverings of the brain and spinal cord.",
    "Pituitary": "Adenoma developing in the pituitary gland at the base of the brain (sella turcica), often impacting hormone regulation and vision.",
    "No Tumor": "Normal neuroimaging scan with no focal mass, hemorrhage, or midline shift detected."
}

# Visual Branding Palette
CLASS_COLORS = {
    "Glioma": "#E63946",      # Vibrant Crimson
    "Meningioma": "#F4A261",  # Amber Ochre
    "No Tumor": "#2A9D8F",    # Medical Teal / Emerald
    "Pituitary": "#457B9D"    # Slate Sapphire
}

# Model Filenames & Configurations
XCEPTION_WEIGHTS_FILE = MODELS_DIR / "xception_model.weights.h5"
CNN_MODEL_FILE = MODELS_DIR / "cnn_model.h5"
CNN_KERAS_FILE = MODELS_DIR / "cnn_model.keras"

INPUT_SHAPES = {
    "xception": (299, 299, 3),
    "cnn": (224, 224, 3)
}

# Explainable AI (XAI) Parameters
SALIENCY_THRESHOLD_PERCENTILE = 80
GAUSSIAN_BLUR_KERNEL = (11, 11)
HEATMAP_ALPHA = 0.7
ORIGINAL_ALPHA = 0.3

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
