#!/usr/bin/env bash
# =========================================================================
#   CropYield-ST: AI-Based Hyperlocal Crop Yield Forecasting
#   One-Click Double-Clickable Launcher for macOS
#   Course: Artificial Intelligence - B.Tech CSE, Fall Semester 2026-27
# =========================================================================

# Ensure we are in the script's directory
cd "$(cd "$(dirname "$0")" && pwd)"

echo "========================================================================="
echo "  CropYield-ST: AI-Based Hyperlocal Crop Yield Forecasting"
echo "  macOS Universal Zero-Friction Launcher"
echo "========================================================================="
echo ""

# 1. Check if Docker is available for zero-installation container mode
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "[+] Docker detected! You can run in zero-installation container mode."
    echo "[*] Building/Starting CropYield-ST container..."
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose up --build -d
    else
        docker compose up --build -d
    fi
    echo "[+] Container running on http://localhost:8000"
    sleep 2
    open "http://localhost:8000"
    echo "[*] Application opened in default browser. To stop: docker compose down"
    exit 0
fi

# 2. Native Python Mode (Zero Node.js / Zero npm required)
echo "[*] Checking Python 3 on macOS..."
if command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 was not found on your Mac."
    echo "Please install Python 3.10+ via https://www.python.org/downloads/ or run 'xcode-select --install'."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[+] Using: $($PY_CMD --version)"

# Check or create virtual environment for clean macOS execution
if [ ! -d ".venv" ]; then
    echo "[*] Initializing isolated Python virtual environment (.venv)..."
    $PY_CMD -m venv .venv
    source .venv/bin/activate
    echo "[*] Installing Python dependencies (one-time setup)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Launch the unified application
echo "[*] Launching CropYield-ST (Frontend + FastAPI on http://localhost:8000)..."
$PY_CMD run_app.py
