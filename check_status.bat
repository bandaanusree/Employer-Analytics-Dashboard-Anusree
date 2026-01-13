@echo off
echo ========================================
echo Dashboard Status Check
echo ========================================
echo.

echo Checking Backend (Port 5000)...
curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Backend is running
    curl -s http://localhost:5000/api/health
    echo.
) else (
    echo [ERROR] Backend is NOT running
    echo Please start backend: cd backend ^&^& python app.py
    echo.
)

echo Checking Frontend (Port 3000)...
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Frontend is running
) else (
    echo [ERROR] Frontend is NOT running
    echo Please start frontend: cd frontend ^&^& npm run dev
    echo.
)

echo.
echo ========================================
echo Status Check Complete
echo ========================================
echo.
echo If both are running, open: http://localhost:3000
echo.
pause

