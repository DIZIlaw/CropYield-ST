@echo off
title CropYield-ST Docker Launcher
echo =========================================================================
echo   CropYield-ST: Zero-Installation Docker Launcher
echo   Runs without needing Python or Node.js on your computer!
echo =========================================================================
echo.
cd /d "%~dp0"

where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker was not found. Please install Docker Desktop from https://www.docker.com/
    pause
    exit /b 1
)

echo [*] Starting CropYield-ST container...
docker compose up --build -d

echo.
echo [*] Waiting for container to start...
timeout /t 3 /nobreak >nul

echo [*] Opening browser to http://localhost:8000 ...
start http://localhost:8000

echo.
echo [SUCCESS] CropYield-ST is running at http://localhost:8000!
echo To stop the container later, run: docker compose down
echo.
pause
