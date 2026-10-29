#!/usr/bin/env bash
# =========================================================================
#   CropYield-ST: AI-Based Hyperlocal Crop Yield Forecasting
#   macOS / Linux Terminal Launcher
# =========================================================================

cd "$(cd "$(dirname "$0")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
else
    PY_CMD="python"
fi

echo "[*] Launching CropYield-ST using $PY_CMD..."
$PY_CMD run_app.py
