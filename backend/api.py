from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import base64
import io
from PIL import Image

from backend.config import (
    CLASSES,
    CLASS_DESCRIPTIONS,
    CLASS_COLORS,
    SAMPLES_DIR,
    SALIENCY_DIR,
    ROOT_DIR,
    XCEPTION_WEIGHTS_FILE,
    CNN_MODEL_FILE,
    CNN_KERAS_FILE
)
from backend.services.inference import predict_tumor, load_model_instance
from backend.services.saliency import generate_saliency_map_tf, generate_synthetic_saliency_map
from backend.services.gemini_explainer import generate_neurological_explanation

app = FastAPI(
    title="TumorVision Diagnostic API",
    description="Clinical AI API for Brain Tumor Classification, Saliency Mapping, and Gemini Explanations",
    version="2.0.0"
)

# Enable CORS for local web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT_DIR / "frontend"

if SAMPLES_DIR.exists():
    app.mount("/static/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
if SALIENCY_DIR.exists():
    app.mount("/static/saliency", StaticFiles(directory=str(SALIENCY_DIR)), name="saliency")

@app.get("/")
def serve_index():
    """Serves the TumorVision web interface at the root URL."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "TumorVision API is running. Visit /docs for the interactive Swagger documentation."}

@app.get("/style.css")
def serve_css():
    """Serves frontend styling."""
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS not found")

@app.get("/app.js")
def serve_js():
    """Serves frontend client script."""
    js_path = FRONTEND_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS not found")

@app.get("/api/health")
def health_check():
    """Returns system status, active models, and weights availability."""
    return {
        "status": "healthy",
        "classes": CLASSES,
        "weights": {
            "xception_weights_found": XCEPTION_WEIGHTS_FILE.exists(),
            "cnn_model_found": CNN_MODEL_FILE.exists() or CNN_KERAS_FILE.exists()
        },
        "samples_count": len(list(SAMPLES_DIR.glob("*.png")))
    }

@app.get("/api/samples")
def get_sample_scans():
    """Returns metadata for pre-packaged test MRI scans."""
    samples = []
    for file in SAMPLES_DIR.glob("*.png"):
        cls_name = file.stem.replace("_sample", "").capitalize()
        if cls_name.lower() == "notumor":
            cls_name = "No Tumor"
        samples.append({
            "id": file.stem,
            "filename": file.name,
            "label": cls_name,
            "url": f"/static/samples/{file.name}",
            "description": CLASS_DESCRIPTIONS.get(cls_name, "Clinical validation scan.")
        })
    return {"samples": samples}

@app.post("/api/diagnose")
async def diagnose_scan(
    file: UploadFile = File(...),
    model_choice: str = Form("Transfer Learning - Xception"),
    api_key: str = Form(None)
):
    """
    All-in-one diagnostic pipeline:
    1. Computes deep learning classification & class probabilities.
    2. Generates Explainable AI Saliency Map.
    3. Generates Gemini 1.5 Flash Neurologist clinical commentary.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # 1. Prediction
    pred_res = predict_tumor(pil_img, model_type=model_choice)
    pred_class = pred_res["predicted_class"]
    confidence = pred_res["confidence"]
    class_idx = pred_res["class_index"]
    probabilities = pred_res["probabilities"]

    # 2. Saliency Map Generation
    model_instance, has_weights = load_model_instance("xception" if "xception" in model_choice.lower() else "cnn")
    filename = f"diag_{Path(file.filename).stem}.png"
    
    if model_instance is not None and has_weights:
        try:
            saliency_overlay, saliency_path = generate_saliency_map_tf(
                model=model_instance,
                img_array=pred_res["pil_image"],
                class_index=class_idx,
                target_size=pred_res["target_size"],
                original_pil_img=pil_img,
                save_filename=filename
            )
        except Exception:
            saliency_overlay, saliency_path = generate_synthetic_saliency_map(
                original_pil_img=pil_img,
                class_index=class_idx,
                target_size=pred_res["target_size"],
                save_filename=filename
            )
    else:
        saliency_overlay, saliency_path = generate_synthetic_saliency_map(
            original_pil_img=pil_img,
            class_index=class_idx,
            target_size=pred_res["target_size"],
            save_filename=filename
        )

    # Encode saliency map as base64 for direct frontend rendering
    buffered = io.BytesIO()
    Image.fromarray(saliency_overlay).save(buffered, format="PNG")
    saliency_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 3. Neurological explanation
    explanation = generate_neurological_explanation(
        img_path=saliency_path,
        model_prediction=pred_class,
        confidence=confidence,
        api_key=api_key
    )

    return {
        "prediction": {
            "class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "description": pred_res["description"],
            "color": pred_res["color"],
            "model_used": pred_res["model_used"],
            "mode": pred_res["mode"]
        },
        "saliency_map": {
            "image_base64": f"data:image/png;base64,{saliency_b64}",
            "file_path": str(saliency_path)
        },
        "neurologist_assessment": explanation
    }
