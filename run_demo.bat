@echo off
title CropYield-ST One-Click Launcher
echo =========================================================================
echo   CropYield-ST: AI-Based Hyperlocal Crop Yield Forecasting
echo   Multi-Spectral Satellite and SAR Fusion Using Spatio-Temporal GNNs
echo   Course: Artificial Intelligence - B.Tech CSE, Fall Semester 2026-27
echo =========================================================================
echo.
cd /d "%~dp0"

:: Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3 was not found in your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo or ensure Python is added to your Environment Variables.
    echo.
    pause
    exit /b 1
)

echo [*] Launching unified CropYield-ST Full-Stack Application...
echo [*] Opening browser to http://127.0.0.1:8000 ...
python run_app.py
pause
