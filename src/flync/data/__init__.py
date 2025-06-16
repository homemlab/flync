"""Data package for FLYNC.

This package contains the trained models and static data files
required for FLYNC pipeline execution.
"""

from pathlib import Path

# Get the data directory path
DATA_DIR = Path(__file__).parent

# Model paths
MODEL_DIR = DATA_DIR / "model"
STATIC_DIR = DATA_DIR / "static"

# Default model path
DEFAULT_MODEL_PATH = MODEL_DIR / "rf_dm6_lncrna_classifier.model"

# Static data files
TRACKS_FILE = STATIC_DIR / "tracksFile.tsv"
REQUIRED_LINKS_FILE = STATIC_DIR / "required_links.txt"
FLY_CUTOFF_FILE = STATIC_DIR / "fly_cutoff.txt"
FLY_HEXAMER_FILE = STATIC_DIR / "fly_Hexamer.tsv"


def get_model_path() -> Path:
    """Get the path to the default trained model."""
    return DEFAULT_MODEL_PATH


def get_static_file(filename: str) -> Path:
    """Get the path to a static data file."""
    return STATIC_DIR / filename


def get_training_data_path(dataset: str) -> Path:
    """Get the path to training data.
    
    Args:
        dataset: Either 'lncrna' or 'not_lncrna'
    """
    return MODEL_DIR / dataset
