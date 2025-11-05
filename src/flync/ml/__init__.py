"""
Model utilities for flync EBM classifier.

This module provides schema validation and prediction utilities for the
EBM-based lncRNA classification model.
"""

from .ebm_predictor import EBMPredictor, EBMPredictorWithScaler
from .schema_extractor import (
    ModelSchema,
    extract_and_save_schema,
    extract_schema_from_model,
)
from .schema_validator import (
    SchemaValidator,
    ValidationIssue,
    ValidationLevel,
    ValidationMode,
    ValidationResult,
    validate_dataframe,
)

__all__ = [
    # Predictor
    "EBMPredictor",
    "EBMPredictorWithScaler",
    # Schema extraction
    "ModelSchema",
    "extract_schema_from_model",
    "extract_and_save_schema",
    # Validation
    "SchemaValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "ValidationMode",
    "validate_dataframe",
]
