"""
CropYield-ST: Standalone One-Click Desktop Launcher
Launches the full-stack CropYield-ST application (FastAPI + React Dashboard)
and opens the default web browser to http://127.0.0.1:8000.
"""
import os
import sys
import time
import subprocess
import webbrowser

def get_base_dir():
    """Gets the base directory whether running as a PyInstaller executable or script."""
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def main():
    base_dir = get_base_dir()
    os.chdir(base_dir)

    print("=" * 72)
    print("  CropYield-ST: Hyperlocal Crop Yield Forecasting Dashboard")
    print("  One-Click Application Launcher")
    print("=" * 72)
    print(f"[*] Base directory: {base_dir}")

    # Path to run_app.py
    run_app_path = os.path.join(base_dir, "run_app.py")
    if not os.path.exists(run_app_path):
        print(f"[ERROR] Could not find run_app.py at {run_app_path}")
        input("Press Enter to exit...")
        return

    # Use system python
    python_exe = sys.executable if not getattr(sys, 'frozen', False) else "python"
    
    print("[*] Starting CropYield-ST unified server on http://127.0.0.1:8000 ...")
    
    # Launch run_app.py
    try:
        proc = subprocess.Popen([python_exe, run_app_path], cwd=base_dir)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    except Exception as e:
        print(f"[!] Launch error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
