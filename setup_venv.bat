@echo off
echo ==============================================
echo Setting up Virtual Environment for Corvinius
echo ==============================================

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.9 - 3.11.
    pause
    exit /b 1
)

:: Create virtual environment
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

:: Upgrade pip and install requirements
echo Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo Setup completed successfully!
echo To run the simulation, run:
echo   .venv\Scripts\python.exe simulation.py
echo ==============================================
pause
