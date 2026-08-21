import os
import sys
import webbrowser
from pathlib import Path

# Add root and backend to python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
MODULES_DIR = BACKEND_DIR / "modules"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(MODULES_DIR))

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print(" Launching Product Review Classifier React Dashboard")
    print("=" * 70)
    print(" - Backend API: http://127.0.0.1:8000/api/health")
    print(" - Frontend Dashboard: http://127.0.0.1:8000/")
    print("=" * 70)

    # Open browser automatically after a short delay
    webbrowser.open("http://127.0.0.1:8000/")

    # Start FastAPI server
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
