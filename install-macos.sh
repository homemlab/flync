#!/bin/bash
# FLYNC Dependency Installation Script for macOS

set -e

echo "🚀 Installing FLYNC dependencies on macOS..."

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Update Homebrew
echo "📦 Updating Homebrew..."
brew update

# Install system dependencies
echo "🔧 Installing system tools..."
brew install wget curl gzip

# Install Python if not present
echo "🐍 Installing Python..."
brew install python@3.11

# Install R
echo "📊 Installing R..."
brew install r

# Install bioinformatics tools
echo "🧬 Installing bioinformatics tools..."
brew install samtools bedtools

# Install conda/mamba for better bioinformatics tool management
if ! command -v conda &> /dev/null; then
    echo "🐍 Installing Miniconda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-MacOSX-x86_64.sh
    
    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bash_profile
    source ~/.bash_profile
    
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
