#!/usr/bin/env python3
"""Simple test runner for FLYNC installation."""

import sys
from pathlib import Path

# Add src to path so we can import flync
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Run installation test
from tests.test_installation import *

if __name__ == "__main__":
    print("Running FLYNC installation test...")
    
    try:
        test_basic_imports()
        test_package_structure()
        deps_ok = test_optional_dependencies()
        
        if deps_ok:
            print("\n🎉 FLYNC installation test PASSED!")
            print("\nYou can now run:")
            print("  python -m flync.cli --help")
            print("  python -m flync.cli init-config")
        else:
            print("\n⚠️  Installation test completed with warnings.")
            print("   Install missing dependencies with: pip install -r requirements.txt")
    
    except Exception as e:
        print(f"\n❌ Installation test FAILED: {e}")
        sys.exit(1)
