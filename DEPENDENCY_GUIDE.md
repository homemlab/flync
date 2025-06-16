# FLYNC Dependency Management Guide

## 🎯 Quick Start

### 1. Check Current Status
```bash
python -m flync check-deps
```
This will show you exactly what's installed and what's missing.

### 2. One-Command Installation
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate flync
pip install -e .

# Or use platform-specific scripts
bash setup.sh  # Interactive setup assistant
```

### 3. Verify Installation
```bash
python -m flync check-deps
```

## 🛠️ Management Commands

### Check Dependencies
```bash
# Full dependency check with installation instructions
python -m flync check-deps

# Quick status check (used internally by pipeline)
python -c "from src.flync.utils.dependencies import DependencyManager; print('Ready' if DependencyManager().is_ready_for_pipeline() else 'Missing deps')"
```

### Generate Installation Commands
```bash
# Show conda commands (default)
python -m flync install-deps --dry-run

# Show apt commands (Ubuntu)
python -m flync install-deps --pm apt --dry-run

# Show brew commands (macOS)  
python -m flync install-deps --pm brew --dry-run
```

### Pipeline Integration
The pipeline automatically checks dependencies before running:
```bash
# Runs dependency check first
python -m flync run --config config.yml

# Skip dependency check (not recommended)
python -m flync run --config config.yml --skip-deps
```

## 📁 Files Overview

- `environment.yml` - Complete conda environment specification
- `setup.sh` - Interactive setup assistant (cross-platform)
- `install-ubuntu.sh` - Ubuntu/Debian automated installer
- `install-macos.sh` - macOS automated installer  
- `install-windows.ps1` - Windows PowerShell installer
- `src/flync/utils/dependencies.py` - Dependency management system

## 🔧 Troubleshooting

### Common Issues

**"Tool not found in PATH"**
- Make sure conda environment is activated: `conda activate flync`
- Verify installation: `which toolname` or `toolname --version`

**"Permission denied"**
- Some installers need admin privileges
- Use `sudo` on Linux/macOS or "Run as Administrator" on Windows

**"Conda environment conflicts"**
- Try using mamba instead: `mamba env create -f environment.yml`
- Or create a clean environment: `conda create -n flync-clean python=3.11`

### Manual Installation

If automated scripts fail, install tools individually:

```bash
# Core bioinformatics tools
conda install -c bioconda hisat2 stringtie gffread samtools bedtools

# Python tool via pip
pip install CPAT

# R packages
conda install -c conda-forge r-ballgown r-genefilter
```

## 🎯 Platform-Specific Notes

### Linux (Ubuntu/Debian)
- Use `install-ubuntu.sh` for automated setup
- Most tools available via apt + conda
- Best compatibility

### macOS  
- Use `install-macos.sh` for automated setup
- Some tools need Homebrew + conda combination
- Generally good compatibility

### Windows
- **Recommended**: Use WSL2 + Ubuntu setup
- **Alternative**: Docker container
- **Advanced**: Native tools via conda (limited compatibility)

## ✅ Success Checklist

After installation, you should see:
```bash
$ python -m flync check-deps
✅ All required tools available
✅ All optional tools available
✅ Ready to run FLYNC pipeline!
```

If you see any ❌ or ⚠️ symbols, follow the provided installation commands.

## 🆘 Getting Help

1. Check this guide and `SYSTEM_DEPENDENCIES.md`
2. Run `python -m flync check-deps` for specific missing tools
3. Try the interactive setup: `bash setup.sh`
4. For issues, check the GitHub issues or create a new one

Remember: The dependency checker is your friend! Use it whenever you're unsure about your setup.
