#!/usr/bin/env bash
# =====================================================================
# Quantitative Finance Paper Scraper & Knowledge Engine Launcher (Bash)
# =====================================================================

set -e

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo " Quantitative Finance Paper Scraper & Knowledge Engine Launcher"
echo "====================================================================="
echo ""

# 1. Detect Python 3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 was not found in PATH."
    echo "Please install Python 3.10+ (e.g., sudo apt install python3 python3-venv)"
    exit 1
fi

echo "[OK] Found $($PYTHON_CMD --version)"

# 2. Virtual Environment Setup
if [ ! -f "venv/bin/activate" ]; then
    echo "[SETUP] Virtual environment not found. Creating 'venv'..."
    $PYTHON_CMD -m venv venv
    echo "[OK] Virtual environment created."
fi

# 3. Activate Virtual Environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 4. Check and Install Dependencies
echo "[INFO] Verifying requirements.txt..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# 5. Check .env Configuration
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[SECURITY] Note: You do NOT need to save secrets on disk. You can enter them securely in memory during launch."
    fi
fi

# 6. Launch Application
echo "[INFO] Launching application..."
if [ $# -eq 0 ]; then
    echo "[INFO] Launching Interactive Terminal Menu..."
    python main.py
else
    echo "[INFO] Passing arguments: $@"
    python main.py "$@"
fi
