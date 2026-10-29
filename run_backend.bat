@echo off
title CropYield-ST - FastAPI Backend Service
echo ======================================================
echo   Starting CropYield-ST Backend Service on port 8000...
echo ======================================================
cd /d "%~dp0"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
pause
