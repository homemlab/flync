"""Utility functions for FLYNC pipeline."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import requests


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def check_command_availability(command: str) -> bool:
    """Check if a command is available in PATH.
    
    Args:
        command: Command name to check
        
    Returns:
        True if command is available, False otherwise
    """
    return shutil.which(command) is not None


def run_command_safe(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """Run a command safely with proper error handling.
    
    Args:
        command: Command and arguments
        cwd: Working directory
        timeout: Timeout in seconds
        
    Returns:
        CompletedProcess object
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out: {' '.join(command)}")
    except Exception as e:
        raise RuntimeError(f"Error running command: {e}")


def validate_file_exists(file_path: Path, description: str = "File") -> None:
    """Validate that a file exists, raise error if not.
    
    Args:
        file_path: Path to check
        description: Description for error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")


def create_directory_structure(base_dir: Path, subdirs: List[str]) -> None:
    """Create a directory structure.
    
    Args:
        base_dir: Base directory
        subdirs: List of subdirectory names to create
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


def count_lines_in_file(file_path: Path) -> int:
    """Count lines in a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Number of lines
    """
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return file_path.stat().st_size / (1024 * 1024)


def compress_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Compress a file using gzip.
    
    Args:
        input_path: Input file path
        output_path: Output file path (defaults to input_path.gz)
        
    Returns:
        Path to compressed file
    """
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + '.gz')
    
    import gzip
    
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            f_out.writelines(f_in)
    
    return output_path


def decompress_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Decompress a gzip file.
    
    Args:
        input_path: Input compressed file path
        output_path: Output file path (defaults to input without .gz)
        
    Returns:
        Path to decompressed file
    """
    if output_path is None:
        if input_path.suffix == '.gz':
            output_path = input_path.with_suffix('')
        else:
            output_path = input_path.with_suffix('.decompressed')
    
    import gzip
    
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.writelines(f_in)
    
    return output_path
