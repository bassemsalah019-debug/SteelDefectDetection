@echo off
REM SteelVision frontend (Vite dev server) -> http://localhost:5173
title SteelVision Frontend
cd /d "%~dp0frontend"
echo Starting SteelVision frontend on http://localhost:5173 ...
call npm run dev
pause
