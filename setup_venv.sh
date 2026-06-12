#!/bin/bash
set -e

echo "=============================================="
echo "Setting up Virtual Environment for Corvinius"
echo "=============================================="

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed. Please install Python 3.9 - 3.11."
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Upgrade pip and install requirements
echo "Installing dependencies from requirements.txt..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup completed successfully!"
echo "To run the simulation, run:"
echo "  .venv/bin/python simulation.py"
echo "=============================================="
