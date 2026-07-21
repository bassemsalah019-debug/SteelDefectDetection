@echo off
REM ============================================================
REM  SteelVision - start the WHOLE app (backend + frontend)
REM  Double-click this file. Two windows open; the browser
REM  opens at http://localhost:5173 after a few seconds.
REM ============================================================
cd /d "%~dp0"
start "SteelVision API" cmd /k "%~dp0start-backend.bat"
start "SteelVision Frontend" cmd /k "%~dp0start-frontend.bat"
echo Backend : http://localhost:8000   (API docs: http://localhost:8000/docs)
echo Frontend: http://localhost:5173
echo Opening the app in your browser...
timeout /t 8 >nul
start http://localhost:5173
