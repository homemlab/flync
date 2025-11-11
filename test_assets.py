#!/usr/bin/env python
"""Test that all bundled assets are accessible after installation.

This script verifies that the FLYNC package includes all required data files.
Run after installing the package to ensure assets are properly bundled.
"""

import sys
from pathlib import Path


def test_assets():
    """Verify all asset files are accessible."""
    try:
        import flync

        package_dir = Path(flync.__file__).parent

        # Test model assets
        assets_dir = package_dir / "assets"
        required_assets = [
            "flync_ebm_model.pkl",
            "flync_ebm_scaler.pkl",
            "flync_ebm_model_schema.json",
            "flync_ebm_model_params.json",
        ]

        print("Checking assets directory:", assets_dir)
        missing_assets = []
        for asset in required_assets:
            asset_path = assets_dir / asset
            if asset_path.exists():
                print(f"  ✓ {asset} ({asset_path.stat().st_size:,} bytes)")
            else:
                print(f"  ✗ {asset} (missing)")
                missing_assets.append(asset)

        # Test config assets
        config_dir = package_dir / "config"
        required_configs = ["bwq_config.yaml"]

        print("\nChecking config directory:", config_dir)
        missing_configs = []
        for config in required_configs:
            config_path = config_dir / config
            if config_path.exists():
                print(f"  ✓ {config} ({config_path.stat().st_size:,} bytes)")
            else:
                print(f"  ✗ {config} (missing)")
                missing_configs.append(config)

        # Test workflow assets
        workflows_dir = package_dir / "workflows"
        required_workflows = ["Snakefile"]

        print("\nChecking workflows directory:", workflows_dir)
        missing_workflows = []
        for workflow in required_workflows:
            workflow_path = workflows_dir / workflow
            if workflow_path.exists():
                print(f"  ✓ {workflow} ({workflow_path.stat().st_size:,} bytes)")
            else:
                print(f"  ✗ {workflow} (missing)")
                missing_workflows.append(workflow)

        # Summary
        all_missing = missing_assets + missing_configs + missing_workflows
        if all_missing:
            print(f"\n✗ Test failed: {len(all_missing)} files missing")
            print("Missing files:", ", ".join(all_missing))
            return False
        else:
            print("\n✓ All required assets are present")
            return True

    except ImportError as e:
        print(f"✗ Failed to import flync: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_assets()
    sys.exit(0 if success else 1)
