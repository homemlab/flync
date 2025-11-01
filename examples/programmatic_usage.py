"""
Example: Using FLYNC Programmatically
=====================================

This example shows how to use FLYNC components from Python code
instead of the command line.
"""

from pathlib import Path
import yaml

# Example 1: Load configuration
def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Example 2: Access package version
def get_version():
    """Get FLYNC version."""
    from flync import __version__
    return __version__

# Example 3: Use ML prediction (conceptual - requires adapted code)
def run_ml_prediction(feature_file, model_path, output_dir):
    """
    Run ML prediction on features.
    
    Note: This is a conceptual example. The actual ML scripts
    would need to be adapted to work as importable modules.
    """
    # This would require refactoring the ML scripts to use functions
    # instead of direct script execution
    pass

# Example 4: Access model and data paths
def get_package_paths():
    """Get important package paths."""
    import flync
    package_dir = Path(flync.__file__).parent.parent.parent
    
    return {
        'package': package_dir,
        'model': package_dir / 'model',
        'scripts': package_dir / 'scripts',
        'static': package_dir / 'static',
    }

# Example usage
if __name__ == '__main__':
    print(f"FLYNC version: {get_version()}")
    
    paths = get_package_paths()
    print(f"\nPackage paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Load example config
    config_path = paths['package'] / 'test' / 'config.yaml'
    if config_path.exists():
        config = load_config(config_path)
        print(f"\nExample config loaded:")
        print(f"  Output: {config.get('output')}")
        print(f"  Threads: {config.get('threads')}")
