"""
CropYield-ST: Universal Cross-Platform Application Launcher
Runs on Windows, macOS, and Linux without requiring Node.js or npm.
Serves the unified React Frontend and FastAPI Backend on http://127.0.0.1:8000.
"""
import os
import sys
import time
import threading
import webbrowser
import uvicorn

# Ensure the project root is in Python module search path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def open_browser(url: str, delay_seconds: float = 1.5):
    """Opens the user's default browser after the server has initialized."""
    time.sleep(delay_seconds)
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[*] Note: Open your browser to {url} ({e})")


def check_and_prepare_data():
    """Ensures deterministic sample datasets exist before serving."""
    sample_dir = os.path.join(PROJECT_ROOT, "data", "sample")
    geojson_path = os.path.join(sample_dir, "parcels.geojson")
    ts_path = os.path.join(sample_dir, "timeseries.csv")
    labels_path = os.path.join(sample_dir, "yield_labels.csv")

    if not (os.path.exists(geojson_path) and os.path.exists(ts_path) and os.path.exists(labels_path)):
        print("[*] Generating calibrated agricultural demonstration dataset...")
        try:
            from backend.app.data.sample_generator import export_sample_dataset
            export_sample_dataset(output_dir=sample_dir)
            print("[+] Sample dataset ready in data/sample/.")
        except Exception as e:
            print(f"[!] Warning during dataset generation: {e}")


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

    print("=" * 72)
    print("  CropYield-ST: Hyperlocal Crop Yield Forecasting Dashboard")
    print("  Multi-Spectral Satellite & SAR Fusion with Spatio-Temporal GNNs")
    print("  Course: Artificial Intelligence — B.Tech CSE, Fall 2026-27")
    print("=" * 72)
    print(f"[*] Project root : {PROJECT_ROOT}")
    print(f"[*] System OS    : {sys.platform}")
    print(f"[*] Python       : {sys.version.split()[0]}")
    print("[*] Initializing unified server (React Frontend + FastAPI REST API)...")

    # Verify sample dataset
    check_and_prepare_data()

    # Pre-build frontend check
    frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
    if os.path.exists(frontend_dist) and os.path.exists(os.path.join(frontend_dist, "index.html")):
        print(f"[+] Found pre-compiled React distribution at: {frontend_dist}")
        print(f"[+] Unified Single-Port Mode: Dashboard & API active at {app_url}")
    else:
        print("[!] Pre-compiled frontend not found in frontend/dist. API will run in headless mode.")
        print("    Run 'npm run build' inside frontend/ to compile the React dashboard.")

    print(f"\n[>>>] Starting CropYield-ST on {app_url}")
    print(f"[>>>] Interactive API Docs : {app_url}/docs")
    print("[>>>] Press Ctrl+C in this terminal to shut down the server.\n")

    # Open browser in a background thread
    threading.Thread(target=open_browser, args=(app_url, 1.8), daemon=True).start()

    # Run uvicorn web server
    try:
        from backend.app.main import app
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n[*] CropYield-ST server stopped gracefully. Goodbye!")
    except Exception as e:
        print(f"\n[!] Server error: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
