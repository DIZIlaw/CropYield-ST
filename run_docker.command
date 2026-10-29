#!/usr/bin/env bash
# =========================================================================
#   CropYield-ST: Zero-Installation Docker Launcher for macOS
# =========================================================================

cd "$(cd "$(dirname "$0")" && pwd)"

echo "========================================================================="
echo "  CropYield-ST: Zero-Installation macOS Launcher"
echo "  Requires no Python or Node.js on your Mac!"
echo "========================================================================="
echo ""

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker was not found."
    echo "Please install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[*] Launching CropYield-ST in Docker..."
if command -v docker-compose >/dev/null 2>&1; then
    docker-compose up --build -d
else
    docker compose up --build -d
fi

echo "[*] Waiting for server..."
sleep 3

echo "[*] Opening browser to http://localhost:8000 ..."
open "http://localhost:8000"

echo ""
echo "[SUCCESS] CropYield-ST is running at http://localhost:8000!"
echo "To stop the container later, run: docker compose down"
echo ""
read -p "Press Enter to close this window..."
