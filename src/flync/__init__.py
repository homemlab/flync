"""
FLYNC: lncRNA discovery pipeline for Drosophila melanogaster

A bioinformatics pipeline for discovering and classifying non-coding genes.
Combines RNA-seq processing, feature extraction from genomic databases,
and machine learning prediction.
"""

__version__ = "1.0.0"
__author__ = "FLYNC Contributors"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("flync")
except PackageNotFoundError:
    # Package is not installed
    pass

# Public API
from flync.api import (
    run_pipeline,
    run_bioinformatics,
    run_ml_prediction,
)

__all__ = [
    "__version__",
    "__author__",
    "run_pipeline",
    "run_bioinformatics",
    "run_ml_prediction",
]
