#!/bin/bash
# FLYNC Dependency Installation Script for Ubuntu/Debian

set -e

echo "🚀 Installing FLYNC dependencies on Ubuntu/Debian..."

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install system dependencies
echo "🔧 Installing system tools..."
sudo apt install -y wget curl gzip build-essential

# Install Python and pip if not present
echo "🐍 Ensuring Python 3.8+ is installed..."
sudo apt install -y python3 python3-pip python3-venv

# Install R
echo "📊 Installing R..."
sudo apt install -y r-base

# Install bioinformatics tools via apt (if available)
echo "🧬 Installing bioinformatics tools..."
sudo apt install -y samtools bedtools

# Install SRA toolkit
echo "📥 Installing SRA toolkit..."
sudo apt install -y sra-toolkit

# Install conda/mamba for better bioinformatics tool management
if ! command -v conda &> /dev/null; then
    echo "🐍 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
    # Initialize conda
    conda init bash
fi

# Install mamba for faster package resolution
echo "⚡ Installing mamba..."
conda install -y mamba -c conda-forge

# Create FLYNC conda environment
echo "🏗️ Creating FLYNC conda environment..."
mamba env create -f environment.yml

echo "✅ Installation complete!"
echo ""
echo "To activate the FLYNC environment:"
echo "  conda activate flync"
echo ""
echo "To install FLYNC Python package:"
echo "  conda activate flync"
echo "  pip install -e ."
echo ""
echo "To check dependencies:"
echo "  python -m flync check-deps"
