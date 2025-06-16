#!/bin/bash
# Quick FLYNC setup script - detects platform and guides installation

set -e

echo "🧬 FLYNC Setup Assistant"
echo "========================"

# Detect operating system
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt &> /dev/null; then
        OS="ubuntu"
    elif command -v yum &> /dev/null; then
        OS="centos"
    else
        OS="linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    OS="windows"
fi

echo "🔍 Detected OS: $OS"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found"
    CONDA_AVAILABLE=true
else
    echo "❌ Conda not found"
    CONDA_AVAILABLE=false
fi

echo ""

# Provide platform-specific guidance
case $OS in
    "ubuntu")
        echo "🐧 Ubuntu/Debian detected"
        echo "📋 Recommended installation:"
        echo "  1. Run: bash install-ubuntu.sh"
        echo "  2. Or use conda: conda env create -f environment.yml"
        ;;
    "macos")
        echo "🍎 macOS detected"
        echo "📋 Recommended installation:"
        echo "  1. Run: bash install-macos.sh"
        echo "  2. Or use conda: conda env create -f environment.yml"
        ;;
    "windows")
        echo "🪟 Windows detected"
        echo "📋 Recommended installation:"
        echo "  1. Use WSL2 + Ubuntu: run install-ubuntu.sh in WSL"
        echo "  2. Or use Docker: docker pull your-registry/flync"
        echo "  3. For PowerShell: run install-windows.ps1 as Administrator"
        ;;
    *)
        echo "❓ Unknown OS detected"
        echo "📋 Try these options:"
        echo "  1. Use conda: conda env create -f environment.yml"
        echo "  2. Install tools manually (see SYSTEM_DEPENDENCIES.md)"
        ;;
esac

echo ""

if [ "$CONDA_AVAILABLE" = true ]; then
    echo "🚀 Quick conda setup:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate flync"
    echo "  pip install -e ."
    echo "  python -m flync check-deps"
    echo ""
    
    read -p "Create conda environment now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🏗️ Creating conda environment..."
        conda env create -f environment.yml
        echo ""
        echo "✅ Environment created! To activate:"
        echo "  conda activate flync"
        echo "  pip install -e ."
        echo "  python -m flync check-deps"
    fi
else
    echo "💡 Install conda/mamba first for easiest setup:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
fi

echo ""
echo "📚 For detailed instructions, see:"
echo "  - SYSTEM_DEPENDENCIES.md"
echo "  - README.md"
