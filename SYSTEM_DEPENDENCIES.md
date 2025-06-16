# FLYNC System Dependencies Management

## Quick Start - Automated Installation

### 🚀 One-Command Setup

**Linux (Ubuntu/Debian):**
```bash
curl -O https://raw.githubusercontent.com/yourrepo/flync/main/install-ubuntu.sh
bash install-ubuntu.sh
```

**macOS:**
```bash
curl -O https://raw.githubusercontent.com/yourrepo/flync/main/install-macos.sh
bash install-macos.sh
```

**Windows:**
```powershell
# Run as Administrator
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/yourrepo/flync/main/install-windows.ps1" -OutFile "install-windows.ps1"
.\install-windows.ps1
```

### 🐍 Using Conda (Recommended)

If you already have conda/mamba installed:
```bash
# Create environment from file
conda env create -f environment.yml
conda activate flync

# Install FLYNC package
pip install -e .

# Check dependencies
python -m flync check-deps
```

## Dependency Checking

### Built-in Dependency Checker

FLYNC includes a comprehensive dependency checker:

```bash
# Check all system dependencies
python -m flync check-deps

# Show installation commands (dry run)
python -m flync install-deps --dry-run

# Generate commands for specific package manager
python -m flync install-deps --pm conda --dry-run
python -m flync install-deps --pm apt --dry-run
python -m flync install-deps --pm brew --dry-run
```

The checker will:
- ✅ Verify all required tools are installed
- 📊 Display versions of detected tools  
- 📋 Show installation commands for missing tools
- ⚠️ Warn about optional but recommended tools

## Installation Methods

### Ubuntu/Debian
```bash
# Core bioinformatics tools
sudo apt-get update
sudo apt-get install hisat2 stringtie samtools bedtools

# CPAT (via pip)
pip install CPAT

# SRA toolkit
sudo apt-get install sra-toolkit

# R and packages
sudo apt-get install r-base
R -e "install.packages('BiocManager'); BiocManager::install(c('ballgown', 'genefilter'))"
```

### Conda/Mamba (Recommended)
```bash
# Install all tools in one environment
mamba create -n flync hisat2 stringtie cpat samtools bedtools sra-tools r-ballgown r-genefilter
conda activate flync
```

### macOS (via Homebrew)
```bash
# Install Homebrew tools
brew install hisat2 stringtie samtools bedtools

# Other tools via conda as above
```

### Windows (via WSL or Docker)
- **Recommended**: Use Docker container or WSL2 with Linux tools
- **Alternative**: Install tools individually (more complex)

## Python Package Dependencies

The Python package itself requires these packages (auto-installed with pip):
- click, pydantic, pyyaml, pandas, numpy, scikit-learn, rich, typer, requests
- biopython (Linux/macOS) - for biological file parsing
- pysam (Linux/macOS, optional) - for BAM/SAM handling

## Summary

- **Python packages**: Handled automatically by pip
- **External tools**: Must be installed separately via package manager
- **Recommended approach**: Use conda/mamba for easy tool management
- **Windows users**: Use Docker or WSL2 for best compatibility
