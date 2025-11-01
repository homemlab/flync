# FLYNC Migration Guide

## Overview

FLYNC has been refactored from a collection of shell scripts into a modern Python package. This guide helps you migrate from the old installation method to the new one.

## What Changed?

### Before (v1.5.9 and earlier)
- Multiple separate conda environments (7 total)
- Shell script-based CLI wrapper (`./flync`)
- Manual environment activation required
- Installation via `./conda-env` script

### After (v1.5.9+)
- Single unified conda environment
- Proper Python package with Click-based CLI
- Global `flync` command after installation
- Modern installation via `pip install -e .`

## Migration Steps

### For New Installations

```bash
git clone https://github.com/homemlab/flync.git
cd flync
./install.sh
```

This will:
1. Create a unified conda environment named `flync`
2. Install all dependencies
3. Install FLYNC as a Python package
4. Make the `flync` command available globally

### For Existing Users

If you previously used the old installation method:

1. **Remove old environments** (optional):
   ```bash
   conda env remove -n infoMod
   conda env remove -n mapMod
   conda env remove -n assembleMod
   conda env remove -n codMod
   conda env remove -n dgeMod
   conda env remove -n featureMod
   conda env remove -n predictMod
   ```

2. **Install using new method**:
   ```bash
   cd flync  # your existing clone
   git pull  # get latest changes
   ./install.sh
   ```

3. **Update your scripts**:
   - Old: `./flync sra -l list.txt -o output`
   - New: `flync sra -l list.txt -o output` (works from anywhere)

## Command Changes

All commands remain the same, but you no longer need to run from the flync directory:

```bash
# Activate environment
conda activate flync

# Run commands (from anywhere)
flync --help
flync run -c config.yaml
flync sra -l samples.txt -o results
flync fastq -f /path/to/fastq -o results
```

## Backward Compatibility

The new package is **fully backward compatible**:
- Old `./flync` script still works (though deprecated)
- Old `./conda-env` installation still works (with warning)
- All shell scripts in `scripts/` remain functional
- `parallel.sh` is still used internally

## Benefits of the New Structure

1. **Simpler Installation**: One environment instead of seven
2. **Better Package Management**: Standard Python packaging with pip
3. **Global Access**: Run `flync` from anywhere after installation
4. **Improved CLI**: Better help messages and error handling
5. **Distribution Ready**: Can be published to PyPI in the future
6. **Docker Compatible**: Updated Dockerfile uses the new structure

## Troubleshooting

### "command not found: flync"
Make sure you:
1. Activated the conda environment: `conda activate flync`
2. Installed the package: `pip install -e .`

### "No module named 'flync'"
Reinstall the package:
```bash
cd /path/to/flync
pip install -e .
```

### Legacy Installation Issues
If you prefer the old method, you can still use:
```bash
./conda-env  # Will show deprecation warning
```

## For Docker Users

The Docker image has been updated to use the new structure. Pull the latest image:

```bash
docker pull rfcdsantos/flync:latest
```

Usage remains the same:
```bash
docker run --rm -v $PWD:/data rfcdsantos/flync flync sra -l test/test-list.txt -o /data/output
```

## Questions?

If you encounter issues or have questions:
1. Check the updated README.md
2. Review the package structure in the repository
3. Open an issue on GitHub
