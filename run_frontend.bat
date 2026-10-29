@echo off
title CropYield-ST - React Vite Frontend
echo ======================================================
echo   Starting CropYield-ST Frontend Service on port 5173...
echo ======================================================
cd /d "%~dp0frontend"
npm run dev
pause
