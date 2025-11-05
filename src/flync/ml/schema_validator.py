"""
Schema validation for EBM model inference data.

This module provides comprehensive validation of inference data against
the model's expected schema, ensuring correct feature names, types, and order.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .schema_extractor import ModelSchema


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationMode(Enum):
    """Validation modes for different use cases."""
    STRICT = "strict"  # Fail on any mismatch
    PERMISSIVE = "permissive"  # Auto-fix reorderable issues, warn on extras
    COERCE = "coerce"  # Attempt type conversions


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    level: ValidationLevel
    message: str
    feature: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    
    def __str__(self) -> str:
        parts = [f"[{self.level.value.upper()}]", self.message]
        if self.feature:
            parts.append(f"(feature: {self.feature})")
        if self.expected is not None:
            parts.append(f"expected={self.expected}")
        if self.actual is not None:
            parts.append(f"actual={self.actual}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    data: Optional[pd.DataFrame] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]
    
    @property
    def infos(self) -> List[ValidationIssue]:
        """Get only info-level issues."""
        return [i for i in self.issues if i.level == ValidationLevel.INFO]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"Validation Result: {'PASSED' if self.is_valid else 'FAILED'}")
        lines.append(f"  Errors: {len(self.errors)}")
        lines.append(f"  Warnings: {len(self.warnings)}")
        lines.append(f"  Info: {len(self.infos)}")
        
        if self.errors:
            lines.append("\nErrors:")
            for issue in self.errors:
                lines.append(f"  - {issue}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for issue in self.warnings:
                lines.append(f"  - {issue}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.summary()


class SchemaValidator:
    """
    Validates inference data against model schema.
    
    Parameters
    ----------
    schema : ModelSchema
        The schema to validate against
    mode : ValidationMode, default=ValidationMode.STRICT
        Validation mode controlling behavior
    allow_extra_features : bool, default=False
        Whether to allow extra features in the data (will be dropped)
    check_value_ranges : bool, default=False
        Whether to check if values are within training data ranges
    range_tolerance : float, default=0.1
        Tolerance for range checking (allows values slightly outside training range)
    """
    
    def __init__(
        self,
        schema: ModelSchema,
        mode: ValidationMode = ValidationMode.STRICT,
        allow_extra_features: bool = False,
        check_value_ranges: bool = False,
        range_tolerance: float = 0.1,
    ):
        self.schema = schema
        self.mode = mode
        self.allow_extra_features = allow_extra_features
        self.check_value_ranges = check_value_ranges
        self.range_tolerance = range_tolerance
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data against schema.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to validate
        
        Returns
        -------
        ValidationResult
            Validation result with issues and optionally corrected data
        """
        issues: List[ValidationIssue] = []
        working_data = data.copy()
        
        # 1. Check for missing features
        missing_features = self._check_missing_features(working_data)
        if missing_features:
            for feat in missing_features:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Missing required feature",
                    feature=feat,
                ))
        
        # 2. Check for extra features
        extra_features = self._check_extra_features(working_data)
        if extra_features:
            level = ValidationLevel.WARNING if self.allow_extra_features else ValidationLevel.ERROR
            for feat in extra_features:
                issues.append(ValidationIssue(
                    level=level,
                    message=f"Extra feature not in schema",
                    feature=feat,
                ))
            if self.allow_extra_features:
                working_data = working_data.drop(columns=extra_features)
        
        # 3. Check feature order
        if not missing_features and not (extra_features and not self.allow_extra_features):
            order_issues = self._check_feature_order(working_data)
            if order_issues:
                if self.mode in [ValidationMode.PERMISSIVE, ValidationMode.COERCE]:
                    # Reorder to match schema
                    working_data = working_data[self.schema.feature_names]
                    issues.append(ValidationIssue(
                        level=ValidationLevel.INFO,
                        message=f"Features reordered to match schema",
                    ))
                else:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        message=f"Feature order does not match schema",
                        expected=self.schema.feature_names[:5],
                        actual=list(working_data.columns[:5]),
                    ))
        
        # 4. Check data types
        if not missing_features:
            dtype_issues = self._check_dtypes(working_data)
            if dtype_issues:
                for feat, expected_dtype, actual_dtype in dtype_issues:
                    if self.mode == ValidationMode.COERCE:
                        # Try to coerce
                        try:
                            working_data[feat] = self._coerce_dtype(
                                working_data[feat], expected_dtype
                            )
                            issues.append(ValidationIssue(
                                level=ValidationLevel.INFO,
                                message=f"Type coerced",
                                feature=feat,
                                expected=expected_dtype,
                                actual=str(actual_dtype),
                            ))
                        except Exception as e:
                            issues.append(ValidationIssue(
                                level=ValidationLevel.ERROR,
                                message=f"Type coercion failed: {e}",
                                feature=feat,
                                expected=expected_dtype,
                                actual=str(actual_dtype),
                            ))
                    else:
                        level = ValidationLevel.WARNING if self.mode == ValidationMode.PERMISSIVE else ValidationLevel.ERROR
                        issues.append(ValidationIssue(
                            level=level,
                            message=f"Incorrect data type",
                            feature=feat,
                            expected=expected_dtype,
                            actual=str(actual_dtype),
                        ))
        
        # 5. Check for null values
        null_issues = self._check_nulls(working_data)
        for feat, null_count, null_pct in null_issues:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message=f"Contains {null_count} null values ({null_pct:.2f}%)",
                feature=feat,
            ))
        
        # 6. Check value ranges (if enabled)
        if self.check_value_ranges and self.schema.feature_stats:
            range_issues = self._check_value_ranges(working_data)
            for feat, out_of_range_pct in range_issues:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"{out_of_range_pct:.2f}% of values outside training range",
                    feature=feat,
                ))
        
        # 7. Check boolean feature values
        bool_issues = self._check_boolean_features(working_data)
        for feat, invalid_values in bool_issues:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Boolean feature contains non-boolean values: {invalid_values}",
                feature=feat,
            ))
        
        # Determine if validation passed
        has_errors = any(i.level == ValidationLevel.ERROR for i in issues)
        is_valid = not has_errors
        
        return ValidationResult(
            is_valid=is_valid,
            data=working_data if is_valid or self.mode == ValidationMode.PERMISSIVE else None,
            issues=issues,
        )
    
    def _check_missing_features(self, data: pd.DataFrame) -> List[str]:
        """Check for missing required features."""
        required_features = set(self.schema.feature_names)
        present_features = set(data.columns)
        return list(required_features - present_features)
    
    def _check_extra_features(self, data: pd.DataFrame) -> List[str]:
        """Check for extra features not in schema."""
        required_features = set(self.schema.feature_names)
        present_features = set(data.columns)
        return list(present_features - required_features)
    
    def _check_feature_order(self, data: pd.DataFrame) -> bool:
        """Check if feature order matches schema. Returns True if order is wrong."""
        return list(data.columns) != self.schema.feature_names
    
    def _check_dtypes(self, data: pd.DataFrame) -> List[Tuple[str, str, Any]]:
        """Check if data types match schema. Returns list of (feature, expected, actual)."""
        issues = []
        for feat, expected_dtype in self.schema.feature_dtypes.items():
            if feat not in data.columns:
                continue
            
            actual_dtype = data[feat].dtype
            
            # Normalize dtype comparison
            if not self._dtypes_compatible(actual_dtype, expected_dtype):
                issues.append((feat, expected_dtype, actual_dtype))
        
        return issues
    
    def _dtypes_compatible(self, actual: Any, expected: str) -> bool:
        """Check if actual dtype is compatible with expected dtype string."""
        actual_str = str(actual)
        
        # Exact match
        if actual_str == expected:
            return True
        
        # Float compatibility
        if expected in ["float", "float64", "float32"] and "float" in actual_str:
            return True
        
        # Integer to float is acceptable for continuous features (in permissive mode only)
        # In coerce mode, this will trigger coercion
        if expected in ["float", "float64", "float32"] and "int" in actual_str:
            # Allow in permissive but not strict (will be flagged for coercion)
            return self.mode == ValidationMode.PERMISSIVE
        
        # Boolean compatibility
        if expected == "bool" and actual_str in ["bool", "boolean"]:
            return True
        
        # Object compatibility for nominal
        if expected == "object" and actual_str in ["object", "string"]:
            return True
        
        return False
    
    def _coerce_dtype(self, series: pd.Series, target_dtype: str) -> pd.Series:
        """Attempt to coerce series to target dtype."""
        if target_dtype in ["float", "float64"]:
            return series.astype(np.float64)
        elif target_dtype == "float32":
            return series.astype(np.float32)
        elif target_dtype == "bool":
            return series.astype(bool)
        elif target_dtype in ["int", "int64"]:
            return series.astype(np.int64)
        elif target_dtype == "object":
            return series.astype(object)
        else:
            raise ValueError(f"Unknown target dtype: {target_dtype}")
    
    def _check_nulls(self, data: pd.DataFrame) -> List[Tuple[str, int, float]]:
        """Check for null values. Returns list of (feature, count, percentage)."""
        issues = []
        for feat in self.schema.feature_names:
            if feat not in data.columns:
                continue
            
            null_count = data[feat].isna().sum()
            if null_count > 0:
                null_pct = (null_count / len(data)) * 100
                issues.append((feat, null_count, null_pct))
        
        return issues
    
    def _check_value_ranges(self, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Check if continuous feature values are within training ranges.
        Returns list of (feature, out_of_range_percentage).
        """
        issues = []
        
        for i, feat in enumerate(self.schema.feature_names):
            if feat not in data.columns or feat not in self.schema.feature_stats:
                continue
            
            stats = self.schema.feature_stats[feat]
            if stats.get("feature_type") != "continuous":
                continue
            
            min_val = stats.get("min")
            max_val = stats.get("max")
            
            if min_val is None or max_val is None:
                continue
            
            # Add tolerance
            range_span = max_val - min_val
            tolerance = range_span * self.range_tolerance
            min_threshold = min_val - tolerance
            max_threshold = max_val + tolerance
            
            # Check values
            values = data[feat].dropna()
            out_of_range = ((values < min_threshold) | (values > max_threshold)).sum()
            
            if out_of_range > 0:
                out_of_range_pct = (out_of_range / len(values)) * 100
                issues.append((feat, out_of_range_pct))
        
        return issues
    
    def _check_boolean_features(self, data: pd.DataFrame) -> List[Tuple[str, Set[Any]]]:
        """
        Check that boolean features only contain True/False values.
        Returns list of (feature, invalid_values).
        """
        issues = []
        
        for i, feat in enumerate(self.schema.feature_names):
            if feat not in data.columns:
                continue
            
            feature_type = self.schema.feature_types[i]
            if feature_type != "nominal":
                continue
            
            # Get unique non-null values
            unique_vals = set(data[feat].dropna().unique())
            
            # Check if all values are boolean
            valid_values = {True, False, np.bool_(True), np.bool_(False)}
            invalid_values = unique_vals - valid_values
            
            if invalid_values:
                # Convert to regular Python types for display
                invalid_display = {
                    v if not isinstance(v, (np.bool_, np.integer, np.floating)) 
                    else v.item() 
                    for v in invalid_values
                }
                issues.append((feat, invalid_display))
        
        return issues


def validate_dataframe(
    data: pd.DataFrame,
    schema: ModelSchema,
    mode: ValidationMode = ValidationMode.STRICT,
    allow_extra_features: bool = False,
    check_value_ranges: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate a DataFrame against a schema.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    schema : ModelSchema
        Schema to validate against
    mode : ValidationMode, default=ValidationMode.STRICT
        Validation mode
    allow_extra_features : bool, default=False
        Whether to allow extra features
    check_value_ranges : bool, default=False
        Whether to check value ranges
    
    Returns
    -------
    ValidationResult
        Validation result
    """
    validator = SchemaValidator(
        schema=schema,
        mode=mode,
        allow_extra_features=allow_extra_features,
        check_value_ranges=check_value_ranges,
    )
    return validator.validate(data)
