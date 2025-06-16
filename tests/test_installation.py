"""Test basic installation and imports."""

def test_basic_imports():
    """Test that basic Python imports work."""
    import sys
    import os
    import subprocess
    from pathlib import Path
    
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    print("✓ Basic Python imports work")


def test_package_structure():
    """Test that package structure is correct."""
    from pathlib import Path
    
    # Get the project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    
    # Check key directories exist
    src_dir = project_root / "src" / "flync"
    assert src_dir.exists(), f"Source directory not found: {src_dir}"
    
    # Check key files exist
    key_files = [
        "src/flync/__init__.py",
        "src/flync/cli.py", 
        "src/flync/config.py",
        "pyproject.toml",
        "requirements.txt"
    ]
    
    for file_path in key_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Key file missing: {full_path}"
    
    print("✓ Package structure is correct")


def test_optional_dependencies():
    """Test optional dependencies and warn if missing."""
    optional_deps = [
        ("pydantic", "Configuration validation"),
        ("typer", "CLI interface"),
        ("rich", "Rich console output"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("sklearn", "Machine learning"),
        ("yaml", "Configuration files")
    ]
    
    missing_deps = []
    
    for dep, description in optional_deps:
        try:
            __import__(dep)
            print(f"✓ {dep} available - {description}")
        except ImportError:
            missing_deps.append((dep, description))
            print(f"✗ {dep} missing - {description}")
    
    if missing_deps:
        print("\nTo install missing dependencies:")
        print("pip install -r requirements.txt")
        print("\nOr install individual packages:")
        for dep, _ in missing_deps:
            print(f"pip install {dep}")
    
    return len(missing_deps) == 0


if __name__ == "__main__":
    print("Testing FLYNC installation...")
    
    test_basic_imports()
    test_package_structure() 
    all_deps_available = test_optional_dependencies()
    
    if all_deps_available:
        print("\n✅ All tests passed! FLYNC is ready to use.")
    else:
        print("\n⚠️  Some optional dependencies are missing.")
        print("   Core functionality will work, but install dependencies for full features.")
    
    print("\nTo test the CLI (if dependencies installed):")
    print("python -m flync.cli --help")
