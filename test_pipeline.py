import sys
from pathlib import Path
from PIL import Image

def test_pipeline():
    print("1. Testing config...")
    from backend.config import CLASSES, SAMPLES_DIR, SALIENCY_DIR
    print("Classes:", CLASSES)
    assert len(CLASSES) == 4

    print("\n2. Testing inference...")
    from backend.services.inference import predict_tumor
    test_img_path = SAMPLES_DIR / "pituitary_sample.png"
    pred = predict_tumor(test_img_path)
    print("Predicted:", pred["predicted_class"], "Confidence:", pred["confidence"], "Mode:", pred["mode"])
    assert pred["predicted_class"] in CLASSES

    print("\n3. Testing saliency map generator...")
    from backend.services.saliency import generate_synthetic_saliency_map
    img = Image.open(test_img_path)
    saliency_np, s_path = generate_synthetic_saliency_map(img, pred["class_index"], save_filename="test_saliency.png")
    print("Saliency shape:", saliency_np.shape, "Saved to:", s_path)
    assert Path(s_path).exists()

    print("\n4. Testing Gemini Explainer fallback...")
    from backend.services.gemini_explainer import generate_neurological_explanation
    explanation = generate_neurological_explanation(s_path, pred["predicted_class"], pred["confidence"])
    print("Explanation:", explanation[:100], "...")
    assert len(explanation) > 20

    print("\n5. Testing FastAPI app routes...")
    from backend.api import app
    routes = [r.path for r in app.routes]
    print("API Routes registered:", routes)
    assert "/api/health" in routes
    assert "/api/diagnose" in routes
    assert "/api/samples" in routes

    print("ALL TUMORVISION SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    test_pipeline()
