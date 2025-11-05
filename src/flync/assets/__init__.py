"""
FLYNC Assets Module

This module provides access to packaged model files and configurations.
"""

from pathlib import Path
import importlib.resources as pkg_resources

__all__ = ["get_asset_path", "list_assets"]


def get_asset_path(asset_name: str) -> Path:
    """
    Get the path to a packaged asset file.

    Parameters
    ----------
    asset_name : str
        Name of the asset file

    Returns
    -------
    Path
        Full path to the asset file
    """
    try:
        asset_path = pkg_resources.files("flync.assets").joinpath(asset_name)
        return Path(str(asset_path))
    except Exception as e:
        raise FileNotFoundError(f"Asset '{asset_name}' not found: {e}")


def list_assets() -> list:
    """
    List all available asset files.

    Returns
    -------
    list
        List of asset filenames
    """
    try:
        assets_dir = pkg_resources.files("flync.assets")
        return [f.name for f in Path(str(assets_dir)).iterdir() if f.is_file()]
    except Exception:
        return []
