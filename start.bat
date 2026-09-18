@echo off
setlocal enabledelayedexpansion
title Quantitative Finance Paper Scraper & Discord Bot

echo =====================================================================
echo  Quantitative Finance Paper Scraper & Knowledge Engine Launcher
echo =====================================================================
echo.

:: 1. Verify Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ and ensure "Add Python to PATH" is checked.
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: 2. Check and Create Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Virtual environment not detected. Creating 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP] Virtual environment created successfully.
)

:: 3. Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Verify and Install Dependencies
echo [INFO] Checking dependencies from requirements.txt...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Encountered issues installing some packages. Attempting verbose install...
    python -m pip install -r requirements.txt
)

:: 5. Check Environment Variables File
if not exist ".env" (
    if exist ".env.example" (
        echo [SETUP] No .env file found. Creating from .env.example...
        copy .env.example .env >nul
        echo [ACTION REQUIRED] Created .env template. Please edit .env to add your DISCORD_TOKEN and GLM_API_KEY!
        echo.
    )
)

:: 6. Launch Application
echo [INFO] Starting application...
if "%~1"=="" (
    echo [INFO] No arguments specified. Running default Discord Bot mode (--bot)...
    python main.py --bot
) else (
    echo [INFO] Passing arguments: %*
    python main.py %*
)

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Application exited with code %errorlevel%.
    pause
)
