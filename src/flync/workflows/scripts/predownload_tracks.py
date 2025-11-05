#!/usr/bin/env python3
"""
Pre-download BigWig/BigBed tracks for Docker image

This script pre-downloads all genomic tracks specified in a bwq_config.yaml file
into a persistent cache directory. Used during Docker build to create "prewarmed"
images with tracks already cached.

Usage:
    python predownload_tracks.py <bwq_config.yaml> <cache_dir>
"""

import sys
import yaml
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import requests
from tqdm import tqdm


def compute_url_hash(url: str) -> str:
    """Compute hash for URL to use as cache filename."""
    return hashlib.md5(url.encode()).hexdigest()


def download_track(url: str, cache_dir: Path, verbose: bool = True) -> Path:
    """
    Download a genomic track file to cache directory.

    Parameters
    ----------
    url : str
        URL of the track file
    cache_dir : Path
        Directory to cache the file
    verbose : bool
        Print progress

    Returns
    -------
    Path
        Path to cached file
    """
    # Create filename based on URL hash
    url_hash = compute_url_hash(url)
    cache_file = cache_dir / f"{url_hash}.track"

    # Skip if already cached
    if cache_file.exists():
        if verbose:
            print(f"✓ Already cached: {url}")
        return cache_file

    if verbose:
        print(f"Downloading: {url}")

    try:
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(cache_file, "wb") as f:
            if verbose and total_size > 0:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        if verbose:
            print(f"✓ Downloaded: {cache_file}")

        return cache_file

    except Exception as e:
        # Clean up partial download
        if cache_file.exists():
            cache_file.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")


def predownload_tracks_from_config(
    config_path: Path, cache_dir: Path, verbose: bool = True
):
    """
    Pre-download all tracks specified in a bwq_config.yaml file.

    Parameters
    ----------
    config_path : Path
        Path to bwq_config.yaml file
    cache_dir : Path
        Directory to store cached tracks
    verbose : bool
        Print progress
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("FLYNC Track Pre-downloader")
        print("=" * 60)
        print(f"Config: {config_path}")
        print(f"Cache directory: {cache_dir}")
        print()

    # Load config
    try:
        with open(config_path, "r") as f:
            tracks = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config file: {e}")

    # Validate config structure
    if not isinstance(tracks, list):
        raise ValueError("Config file must contain a list of track definitions")

    if not tracks:
        print("WARNING: No tracks found in config file")
        return

    if verbose:
        print(f"Found {len(tracks)} tracks to download\n")

    # Download each track
    success_count = 0
    failed = []

    for i, track in enumerate(tracks, 1):
        # Get URL from 'path' field (bwq_config uses 'path', not 'url')
        track_url = track.get("path")

        if not track_url:
            print(f"WARNING: Track {i} has no 'path' field, skipping")
            continue

        # Extract a display name from URL
        track_name = track_url.split("/")[-1] if "/" in track_url else track_url

        if verbose:
            print(f"[{i}/{len(tracks)}] {track_name}")

        try:
            download_track(track_url, cache_dir, verbose=verbose)
            success_count += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed.append((track_name, track_url, str(e)))

        if verbose:
            print()

    # Summary
    if verbose:
        print("=" * 60)
        print("Download Summary")
        print("=" * 60)
        print(f"Total tracks: {len(tracks)}")
        print(f"Successfully cached: {success_count}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed downloads:")
            for name, url, error in failed:
                print(f"  - {name}: {error}")

        print()

    if len(failed) > 0:
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python predownload_tracks.py <bwq_config.yaml> <cache_dir>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    cache_dir = Path(sys.argv[2])

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    try:
        predownload_tracks_from_config(config_path, cache_dir, verbose=True)
        print("✓ All tracks pre-downloaded successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
