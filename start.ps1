<#
.SYNOPSIS
    Windows PowerShell startup script for the Quantitative Finance Paper Scraper & Discord Bot.
.DESCRIPTION
    Checks Python installation, sets up a virtual environment, verifies dependencies,
    creates .env if missing, and launches the application.
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Quantitative Finance Paper Scraper & Knowledge Engine Launcher" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Found $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# 2. Virtual Environment Setup
$venvPath = Join-Path $PSScriptRoot "venv"
$venvActivate = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host "[SETUP] Virtual environment not found. Creating 'venv'..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "[OK] Virtual environment created successfully." -ForegroundColor Green
}

# 3. Activate Virtual Environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Cyan
& $venvActivate

# 4. Check & Install Dependencies
Write-Host "[INFO] Checking dependencies from requirements.txt..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r (Join-Path $PSScriptRoot "requirements.txt")

# 5. Check .env Configuration
$envFile = Join-Path $PSScriptRoot ".env"
$envExample = Join-Path $PSScriptRoot ".env.example"

if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Write-Host "[SETUP] .env file not found. Copying from .env.example..." -ForegroundColor Yellow
        Copy-Item -Path $envExample -Destination $envFile
        Write-Host "[ACTION REQUIRED] Created .env template. Please edit .env with your DISCORD_TOKEN and GLM_API_KEY!" -ForegroundColor Magenta
    }
}

# 6. Launch Application
Write-Host "[INFO] Launching application..." -ForegroundColor Green
if ($AppArgs.Count -eq 0) {
    Write-Host "[INFO] Defaulting to Discord Bot mode (--bot)..." -ForegroundColor Gray
    python main.py --bot
} else {
    Write-Host "[INFO] Passing arguments: $AppArgs" -ForegroundColor Gray
    python main.py $AppArgs
}
