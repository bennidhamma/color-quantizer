#!/bin/bash
# Setup script for Color Quantizer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Color Quantizer..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install numpy pillow

echo ""
echo "Setup complete! To run the app:"
echo ""
echo "  source venv/bin/activate"
echo "  python color_quantizer_ui.py"
echo ""
echo "Note: You may need to install tkinter system package:"
echo "  sudo apt-get install python3-tk python3-pil.imagetk"
