import sys
import subprocess
import os
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def check_environment():
    """Validates the local environment and ensures required folders exist."""
    print("=" * 60)
    print("🧠 TUMORVISION CLINICAL AI: SYSTEM CHECK")
    print("=" * 60)

    # 1. Directory Checks
    for d in ["backend/models", "data/sample_scans", "saliency_maps"]:
        p = ROOT_DIR / d
        p.mkdir(parents=True, exist_ok=True)
    print("✓ Workspace directories verified.")

    # 2. Check .env
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        example_env = ROOT_DIR / ".env.example"
        if example_env.exists():
            with open(example_env, "r") as f_src, open(env_file, "w") as f_dst:
                f_dst.write(f_src.read())
            print("✓ Created initial .env configuration file.")
    else:
        print("✓ .env configuration file detected.")
    
    # Check Gemini API Key
    from dotenv import load_dotenv
    load_dotenv(env_file)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key.strip() not in ("", "your_gemini_api_key_here"):
        print("✓ Live Gemini AI Key configured.")
    else:
        print("ℹ Gemini API Key not set -> Serving offline clinical neurological knowledge base.")

    # 3. Model Weights Check
    xception_weights = ROOT_DIR / "backend" / "models" / "xception_model.weights.h5"
    cnn_model = ROOT_DIR / "backend" / "models" / "cnn_model.h5"
    
    if xception_weights.exists():
        print(f"✓ Xception Transfer Learning weights found ({xception_weights.stat().st_size / (1024*1024):.1f} MB).")
    else:
        print("ℹ Xception weights not found in backend/models/ -> Running in Diagnostic Simulation Demo Mode.")
        
    if cnn_model.exists():
        print(f"✓ Custom CNN model found ({cnn_model.stat().st_size / (1024*1024):.1f} MB).")

    print("-" * 60)


def run_streamlit():
    """Launches the modern Streamlit web application."""
    print("🚀 Launching TumorVision Streamlit UI on http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT_DIR / "app.py")]
    subprocess.run(cmd)


def run_api():
    """Launches the FastAPI backend server on port 8000."""
    print("🚀 Launching TumorVision FastAPI REST API on http://127.0.0.1:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="TumorVision Application Launcher")
    parser.add_argument(
        "--mode", 
        choices=["streamlit", "api", "web"], 
        default="streamlit",
        help="Choose 'streamlit' to launch the Streamlit app, 'api' to launch FastAPI, or 'web' to serve the web portal."
    )
    args = parser.parse_args()

    check_environment()

    if args.mode == "streamlit":
        run_streamlit()
    elif args.mode == "api":
        run_api()
    elif args.mode == "web":
        print("🌐 To view the modern Web Portal, launch the API with '--mode api' and open 'frontend/index.html' in your browser.")
        run_api()


if __name__ == "__main__":
    main()
