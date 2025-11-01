#!/bin/bash
# Installation script for FLYNC package

echo "Installing FLYNC..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create conda environment from environment.yml
echo "Creating conda environment..."
conda env create -f "$SCRIPT_DIR/environment.yml"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment"
    exit 1
fi

# Activate environment and install package in editable mode
echo "Installing FLYNC package in editable mode..."
eval "$(conda shell.bash hook)"
conda activate flync

pip install -e "$SCRIPT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install FLYNC package"
    exit 1
fi

echo ""
echo "✓ Installation complete!"
echo ""
echo "To use FLYNC:"
echo "  conda activate flync"
echo "  flync --help"
echo ""
