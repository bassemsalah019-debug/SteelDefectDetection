@echo off
REM ============================================================
REM  SteelVision API - one-click launcher (Windows)
REM  Double-click this file, then open http://localhost:8000/docs
REM ============================================================
title SteelVision API
cd /d "%~dp0backend"

REM Prefer the project venv that already has all dependencies; fall back to PATH python.
set "PY=C:\Users\student\Downloads\files\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting SteelVision API on http://localhost:8000 ...
echo Interactive API docs: http://localhost:8000/docs
echo Press CTRL+C to stop.
echo.
"%PY%" -m uvicorn app.main:app --port 8000
pause
