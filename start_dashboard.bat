@echo off
echo ========================================
echo FutureWorks Dashboard
echo ========================================
echo.
echo Starting Backend and Frontend servers...
echo.
echo Backend will run on: http://localhost:5000
echo Frontend will run on: http://localhost:3000
echo.
echo Opening two windows...
echo.

start "Backend Server" cmd /k "cd backend && python app.py"
timeout /t 3 /nobreak >nul
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo Servers are starting...
echo.
echo Once both servers are running, open your browser to:
echo http://localhost:3000
echo.
pause



