"""Data preparation pipeline for Explainable Boosting Machine models.

This module exposes a reusable API and CLI for cleaning, transforming, and preparing
transcript-level datasets for both model training and inference.

Modes:
    - Training Mode: Cleans, transforms, and splits data for model training
    - Inference Mode: Cleans and transforms data with schema enforcement for predictions
    - Validation Mode: Dry-run to validate data without saving outputs

The inference mode enforces schema consistency using model metadata to ensure all
expected features are present with correct types and values.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

LOGGER = logging.getLogger(__name__)

SCALERS: Dict[str, Optional[Type[StandardScaler]]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


@dataclass
class PipelineConfig:
    """Configuration parameters for the training data preparation pipeline."""

    positive_path: Optional[Path] = None
    negative_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    label_column: str = "y"
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    split_suffix: str = "redux"
    split_output_dir: Path = field(default_factory=lambda: Path("."))
    cleaned_output_path: Optional[Path] = None
    random_seed: int = 99
    scaler: str = "standard"
    enable_multi_hot: bool = True
    prefix_multi_hot: bool = True
    mode: str = "training"  # "training", "inference", or "validation"
    metadata_path: Optional[Path] = None  # Path to model metadata JSON for inference
    scaler_path: Optional[Path] = None  # Path to save/load scaler object
    schema_diff_output: Optional[Path] = None  # Path to save schema diff report


@dataclass
class SchemaDiff:
    """Schema difference report between current data and expected metadata."""

    missing_columns: List[str]
    extra_columns: List[str]
    dtype_mismatches: Dict[str, Tuple[str, str]]  # column: (current, expected)

    def __str__(self) -> str:
        """Format schema diff as human-readable report."""
        lines = ["Schema Difference Report", "=" * 50]

        if self.missing_columns:
            lines.append(f"\nMissing Columns ({len(self.missing_columns)}):")
            for col in sorted(self.missing_columns):
                lines.append(f"  - {col}")
        else:
            lines.append("\nMissing Columns: None")

        if self.extra_columns:
            lines.append(f"\nExtra Columns ({len(self.extra_columns)}):")
            for col in sorted(self.extra_columns):
                lines.append(f"  - {col}")
        else:
            lines.append("\nExtra Columns: None")

        if self.dtype_mismatches:
            lines.append(f"\nData Type Mismatches ({len(self.dtype_mismatches)}):")
            for col, (current, expected) in sorted(self.dtype_mismatches.items()):
                lines.append(f"  - {col}: {current} -> {expected}")
        else:
            lines.append("\nData Type Mismatches: None")

        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Data object capturing the outputs of the preparation pipeline."""

    cleaned_df: pd.DataFrame
    splits: Dict[str, pd.DataFrame]
    cleaned_path: Optional[Path]
    split_paths: Dict[str, Path]
    label_column: Optional[str]
    scaler_path: Optional[Path] = None
    schema_diff: Optional[SchemaDiff] = None


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def execute_data_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full data preparation pipeline using the provided configuration."""

    is_inference = config.mode.lower() == "inference"
    is_validation = config.mode.lower() == "validation"

    LOGGER.info(
        "Starting data preparation mode=%s input_mode=%s suffix=%s scaler=%s multi_hot=%s",
        config.mode,
        "single" if config.dataset_path else "paired",
        config.split_suffix,
        config.scaler,
        config.enable_multi_hot,
    )

    validate_input_paths(config)

    if not is_inference and not is_validation:
        validate_split_sizes(config.train_size, config.val_size, config.test_size)

    if (is_inference or is_validation) and not config.metadata_path:
        raise ValueError(f"metadata_path is required when mode='{config.mode}'.")

    if config.metadata_path and not config.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.metadata_path}")

    df = load_input_dataset(config)

    # In inference/validation mode, the label column may not exist
    label_column = config.label_column
    has_label = label_column in df.columns

    if not is_inference and not is_validation and not has_label:
        raise ValueError(f"Label column '{label_column}' not found in training mode.")

    if has_label:
        df = apply_cleaning_steps(df, label_column)
    else:
        # Apply cleaning without reference to label column
        df = apply_cleaning_steps(df, None)

    # Load scaler if provided (for inference mode)
    loaded_scaler = None
    loaded_scaler_cols = []
    if is_inference and config.scaler_path and config.scaler_path.exists():
        loaded_scaler, loaded_scaler_cols = load_scaler(config.scaler_path)
        LOGGER.info(
            "Loaded scaler from %s (columns=%d)",
            config.scaler_path,
            len(loaded_scaler_cols),
        )

    # Apply transformations WITHOUT scaling (to avoid data leakage)
    df, label_column, _, _ = apply_feature_transformations(
        df,
        label_column=label_column if has_label else None,
        enable_multi_hot=config.enable_multi_hot,
        prefix_multi_hot=config.prefix_multi_hot,
        scaler_name="none"
        if not is_inference
        else config.scaler,  # No scaling in training mode yet
        loaded_scaler=loaded_scaler,
        loaded_scaler_cols=loaded_scaler_cols,
    )

    # Generate schema diff report if in inference/validation mode
    schema_diff = None
    if (is_inference or is_validation) and config.metadata_path:
        schema_diff = generate_schema_diff(df, config.metadata_path, label_column)
        LOGGER.info(
            "Schema diff: missing=%d extra=%d dtype_mismatches=%d",
            len(schema_diff.missing_columns),
            len(schema_diff.extra_columns),
            len(schema_diff.dtype_mismatches),
        )

        # Save schema diff if output path specified
        if config.schema_diff_output:
            save_schema_diff(schema_diff, config.schema_diff_output)

    # In inference/validation mode, enforce schema from metadata
    if is_inference or is_validation:
        df = enforce_schema_from_metadata(df, config.metadata_path)
        LOGGER.info("Enforced schema from metadata file.")

    # Only perform splits in training mode
    if is_inference or is_validation:
        splits = {}
        split_paths = {}
        scaler_obj = None
        scaled_cols = []
    else:
        # Split data FIRST (before scaling)
        splits = perform_splits(
            df,
            label_column=label_column,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            random_seed=config.random_seed,
        )

        # Now fit scaler on training data ONLY and transform all splits
        scaler_obj, scaled_cols = fit_and_apply_scaler_to_splits(
            splits=splits,
            label_column=label_column,
            scaler_name=config.scaler,
        )

        # Reconstruct the full dataframe from scaled splits for saving
        if scaler_obj is not None:
            df = pd.concat(
                [
                    splits["train"],
                    splits.get("val", pd.DataFrame()),
                    splits.get("test", pd.DataFrame()),
                ]
            )
            df = df.sort_index()
            LOGGER.info(
                "Reconstructed full dataset from scaled splits: %d rows", len(df)
            )

    # In validation mode, don't save anything - just report
    if is_validation:
        LOGGER.info(
            "Validation mode: pipeline completed successfully without saving outputs"
        )
        LOGGER.info("Dataset shape: rows=%d cols=%d", df.shape[0], df.shape[1])
        if schema_diff:
            LOGGER.info("\n%s", schema_diff)
        return PipelineResult(
            cleaned_df=df,
            splits=splits,
            cleaned_path=None,
            split_paths={},
            label_column=label_column if has_label else None,
            scaler_path=None,
            schema_diff=schema_diff,
        )

    output_dir = config.split_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_output_path = (
        config.cleaned_output_path
        if config.cleaned_output_path is not None
        else output_dir / f"cleaned_dataset_{config.split_suffix}.parquet"
    )
    cleaned_path = save_dataframe(df, cleaned_output_path)

    # Save scaler in training mode
    scaler_save_path = None
    if not is_inference and scaler_obj is not None and config.scaler_path:
        scaler_save_path = save_scaler(scaler_obj, scaled_cols, config.scaler_path)
        LOGGER.info("Saved scaler to %s", scaler_save_path)

    if not is_inference:
        split_paths = save_splits(splits, output_dir, config.split_suffix)

    LOGGER.info(
        "Pipeline finished mode=%s cleaned_path=%s split_paths=%s",
        config.mode,
        cleaned_path,
        {name: str(path) for name, path in split_paths.items()}
        if not is_inference
        else "N/A",
    )

    return PipelineResult(
        cleaned_df=df,
        splits={name: split.copy() for name, split in splits.items()}
        if not is_inference
        else {},
        cleaned_path=cleaned_path,
        split_paths=split_paths,
        label_column=label_column if has_label else None,
        scaler_path=scaler_save_path,
        schema_diff=schema_diff,
    )


def save_scaler(
    scaler: Any,
    scaled_columns: List[str],
    path: Path,
) -> Path:
    """Save fitted scaler and column names for reproducible inference."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scaler_data = {
        "scaler": scaler,
        "columns": scaled_columns,
        "scaler_type": type(scaler).__name__,
    }

    with open(path, "wb") as f:
        pickle.dump(scaler_data, f)

    LOGGER.info(
        "Saved scaler type=%s columns=%d path=%s",
        scaler_data["scaler_type"],
        len(scaled_columns),
        path,
    )

    return path


def load_scaler(path: Path) -> Tuple[Any, List[str]]:
    """Load fitted scaler and column names from saved file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")

    with open(path, "rb") as f:
        scaler_data = pickle.load(f)

    scaler = scaler_data["scaler"]
    columns = scaler_data["columns"]

    LOGGER.info(
        "Loaded scaler type=%s columns=%d path=%s",
        scaler_data.get("scaler_type", "unknown"),
        len(columns),
        path,
    )

    return scaler, columns


def generate_schema_diff(
    df: pd.DataFrame,
    metadata_path: Path,
    label_column: Optional[str],
) -> SchemaDiff:
    """Generate schema difference report between DataFrame and metadata."""

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    expected_schema = metadata.get("inputs", {})

    # Exclude label column from comparison
    current_cols = set(df.columns)
    if label_column and label_column in current_cols:
        current_cols = current_cols - {label_column}

    expected_cols = set(expected_schema.keys())

    missing_cols = list(expected_cols - current_cols)
    extra_cols = list(current_cols - expected_cols)

    # Check dtype mismatches for common columns
    dtype_mismatches = {}
    common_cols = current_cols & expected_cols

    for col in common_cols:
        current_dtype = str(df[col].dtype)
        expected_dtype = expected_schema[col]

        # Normalize dtype comparison
        if not _dtypes_compatible(current_dtype, expected_dtype):
            dtype_mismatches[col] = (current_dtype, expected_dtype)

    return SchemaDiff(
        missing_columns=missing_cols,
        extra_columns=extra_cols,
        dtype_mismatches=dtype_mismatches,
    )


def _dtypes_compatible(current: str, expected: str) -> bool:
    """Check if two dtypes are compatible."""

    # Exact match
    if current == expected:
        return True

    # Float compatibility
    if "float" in current and "float" in expected:
        return True

    # Int compatibility
    if "int" in current and "int" in expected:
        return True

    # Bool compatibility
    if current == "bool" and expected == "bool":
        return True

    return False


def save_schema_diff(diff: SchemaDiff, path: Path) -> Path:
    """Save schema diff report to file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as text
    with open(path, "w") as f:
        f.write(str(diff))

    # Also save as JSON for programmatic access
    json_path = path.with_suffix(".json")
    diff_dict = {
        "missing_columns": diff.missing_columns,
        "extra_columns": diff.extra_columns,
        "dtype_mismatches": {
            col: {"current": curr, "expected": exp}
            for col, (curr, exp) in diff.dtype_mismatches.items()
        },
    }

    with open(json_path, "w") as f:
        json.dump(diff_dict, f, indent=2)

    LOGGER.info("Saved schema diff report: %s, %s", path, json_path)

    return path


def validate_input_paths(config: PipelineConfig) -> None:
    """Validate mutually exclusive dataset path options and ensure paths exist."""

    if config.dataset_path and (config.positive_path or config.negative_path):
        raise ValueError(
            "Provide either a single dataset path or both positive and negative paths, not both."
        )

    if config.dataset_path is None and not (
        config.positive_path and config.negative_path
    ):
        raise ValueError(
            "Provide either a single dataset path or both positive and negative dataset paths."
        )

    for path in [config.dataset_path, config.positive_path, config.negative_path]:
        if path and not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    """Ensure the split proportions are valid and sum to 1."""

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    for name, size in (("val_size", val_size), ("test_size", test_size)):
        if size < 0 or size >= 1:
            raise ValueError(f"{name} must be between 0 (inclusive) and 1.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0.")


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset from a parquet or CSV file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. Supported formats: parquet, csv."
        )

    # If 'index' column exists (transcript IDs), set it as the DataFrame index
    if "index" in df.columns:
        df = df.set_index("index")
        LOGGER.info(
            "Set 'index' column as DataFrame index (contains %d unique identifiers)",
            df.index.nunique(),
        )

    LOGGER.info(
        "Loaded dataset path=%s rows=%d cols=%d", path, df.shape[0], df.shape[1]
    )
    return df


def load_input_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Load and assemble the dataset according to the configuration."""

    if config.dataset_path:
        df = read_dataset(config.dataset_path)

        # In inference/validation mode, label column is optional
        is_inference = config.mode.lower() in ("inference", "validation")
        has_label = config.label_column in df.columns

        if not is_inference and not has_label:
            raise ValueError(
                f"Label column '{config.label_column}' not found in dataset {config.dataset_path}."
            )

        if has_label:
            df = ensure_boolean_label(df, config.label_column)
            log_label_distribution("loaded_single", df, config.label_column)
        else:
            LOGGER.info(
                "Label column not found (%s mode) - proceeding without labels.",
                config.mode,
            )

        return df

    df_pos = read_dataset(config.positive_path)
    df_neg = read_dataset(config.negative_path)

    df_pos = df_pos.copy()
    df_neg = df_neg.copy()

    df_pos[config.label_column] = True
    df_neg[config.label_column] = False

    if set(df_pos.columns) != set(df_neg.columns):
        missing_in_pos = set(df_neg.columns) - set(df_pos.columns)
        missing_in_neg = set(df_pos.columns) - set(df_neg.columns)
        raise ValueError(
            "Positive and negative datasets must have identical columns after loading. "
            f"missing_in_positive={missing_in_pos} missing_in_negative={missing_in_neg}"
        )

    df = pd.concat([df_pos, df_neg], axis=0, ignore_index=False)
    LOGGER.info(
        "Concatenated positive and negative datasets, preserved index=%s",
        df.index.name or "(unnamed)",
    )
    log_label_distribution("loaded_paired", df, config.label_column)
    return df


def ensure_boolean_label(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Convert the label column to boolean when possible."""

    if df[label_column].dtype == bool:
        return df

    df = df.copy()
    values = df[label_column].dropna().unique()
    if set(values).issubset({0, 1, True, False}):
        df[label_column] = df[label_column].astype(int).astype(bool)
        LOGGER.info("Converted label column '%s' to boolean dtype.", label_column)
    return df


def log_label_distribution(
    stage: str, df: pd.DataFrame, label_column: Optional[str]
) -> None:
    """Log the distribution of the label column."""

    if label_column is None or label_column not in df.columns:
        LOGGER.debug("Label column '%s' missing during stage=%s", label_column, stage)
        return

    distribution = df[label_column].value_counts(dropna=False).to_dict()
    LOGGER.info(
        "Label distribution stage=%s total_rows=%d distribution=%s",
        stage,
        len(df),
        distribution,
    )


def apply_cleaning_steps(df: pd.DataFrame, label_column: Optional[str]) -> pd.DataFrame:
    """Apply data cleaning operations prior to feature transformations."""

    df = df.copy()
    log_label_distribution("pre_cleaning", df, label_column)

    df = add_length_feature(df)
    df = drop_non_feature_columns(df)
    df = handle_missing_values(df)
    df = clean_entries_epdnew(df)

    initial_rows = len(df)
    # Exclude label column from duplicate detection if it exists
    subset_cols = (
        [col for col in df.columns if col != label_column]
        if label_column
        else list(df.columns)
    )
    df = df.drop_duplicates(subset=subset_cols, keep="first")
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed:
        LOGGER.info(
            "Dropped duplicate rows count=%d (index preserved)", duplicates_removed
        )

    log_label_distribution("post_cleaning", df, label_column)
    return df


def add_length_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a transcript length feature when genomic coordinates are available."""

    if {"start", "end"}.issubset(df.columns) and "length" not in df.columns:
        df = df.copy()
        df["length"] = df["end"] - df["start"]
        LOGGER.info("Added length feature using genomic coordinates.")
    return df


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that should not be used as features."""

    columns_to_drop: List[str] = []
    # Drop secondary structure columns except ss_mfe
    columns_to_drop.extend(
        [col for col in df.columns if col.startswith("ss_") and col != "ss_mfe"]
    )
    # Drop genomic coordinate columns to prevent data leakage
    columns_to_drop.extend(
        [col for col in ["chromosome", "start", "end"] if col in df.columns]
    )

    unique_columns = sorted(set(columns_to_drop))
    if unique_columns:
        df = df.drop(columns=unique_columns)
        LOGGER.info("Dropped non-feature columns count=%d", len(unique_columns))
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values according to domain-specific rules."""

    df = df.copy()

    # Fill signal/statistic columns with 0 (missing signal = no signal)
    op1_cols = [
        col
        for col in df.columns
        if col.startswith(("min_", "max_", "mean_", "std_", "sum_", "cov_"))
    ]
    for col in op1_cols:
        missing = df[col].isna().sum()
        if missing:
            df[col] = df[col].fillna(0)
            LOGGER.info("Filled NA values with 0 column=%s count=%d", col, missing)

    op2_cols = [col for col in df.columns if col.startswith("ss_")]
    for col in op2_cols:
        before = len(df)
        if df[col].isna().any():
            df = df.dropna(subset=[col])
            LOGGER.info(
                "Dropped rows with NA values column=%s dropped_rows=%d",
                col,
                before - len(df),
            )
        if col == "ss_mfe":
            before = len(df)
            df = df[df[col] < 0]
            dropped = before - len(df)
            if dropped:
                LOGGER.info("Dropped rows where ss_mfe >= 0 count=%d", dropped)

    op3_cols = [
        col
        for col in df.columns
        if col.startswith(("0", "1")) or "mer_SVD" in col or "mer_svd" in col
    ]
    for col in op3_cols:
        before = len(df)
        if df[col].isna().any():
            df = df.dropna(subset=[col])
            LOGGER.info(
                "Dropped rows with NA values column=%s dropped_rows=%d",
                col,
                before - len(df),
            )

    object_cols = [col for col in df.select_dtypes(include=["object"]).columns]
    for col in object_cols:
        missing = df[col].isna().sum()
        if missing:
            df[col] = df[col].fillna("")
            LOGGER.info(
                "Filled NA values with empty string column=%s count=%d", col, missing
            )

    return df


def clean_entries_epdnew(df: pd.DataFrame) -> pd.DataFrame:
    """Remove promoter ranking suffixes from entries_epdnew values."""

    if "entries_epdnew" not in df.columns:
        return df

    df = df.copy()

    def _sanitize(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        tokens = [
            re.sub(r"_[0-9]+$", "", token.strip())
            for token in text.split(",")
            if token.strip()
        ]
        return ",".join(tokens)

    df["entries_epdnew"] = df["entries_epdnew"].apply(_sanitize)
    LOGGER.info("Sanitized entries_epdnew values to drop promoter rankings.")
    return df


def apply_feature_transformations(
    df: pd.DataFrame,
    *,
    label_column: Optional[str],
    enable_multi_hot: bool,
    prefix_multi_hot: bool,
    scaler_name: str,
    loaded_scaler: Optional[Any] = None,
    loaded_scaler_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str], Optional[Any], List[str]]:
    """Apply feature engineering steps including encoding, sanitisation, and scaling.

    Returns:
        Tuple of (transformed_df, label_column, scaler_object, scaled_columns)
    """

    df = df.copy()

    encoded_df = pd.DataFrame(index=df.index)
    processed_columns: List[str] = []

    if enable_multi_hot:
        df, encoded_df, processed_columns = multi_hot_encode(
            df, label_column, prefix_multi_hot
        )
        if not encoded_df.empty:
            df = pd.concat([df, encoded_df], axis=1)
        LOGGER.info(
            "Applied multi-hot encoding processed_columns=%d generated_features=%d",
            len(processed_columns),
            encoded_df.shape[1] if not encoded_df.empty else 0,
        )
    else:
        LOGGER.info("Skipped multi-hot encoding step (disabled via configuration).")

    df, label_column = sanitize_dataframe_columns(df, label_column)

    # Only reorder label column if it exists
    if label_column and label_column in df.columns:
        target_series = df.pop(label_column)
        df.insert(0, label_column, target_series)

    df = lowercase_and_sort_features(df, label_column)

    # Use loaded scaler for inference, or fit new scaler for training
    if loaded_scaler is not None and loaded_scaler_cols is not None:
        df, scaler_used, scaled_columns = apply_loaded_scaler(
            df, label_column, loaded_scaler, loaded_scaler_cols
        )
        LOGGER.info("Applied loaded scaler to %d features", len(scaled_columns))
    else:
        df, scaler_used, scaled_columns = scale_numeric_features(
            df, label_column, scaler_name
        )
        if scaler_used:
            LOGGER.info(
                "Scaled numeric features scaler=%s feature_count=%d",
                scaler_name,
                len(scaled_columns),
            )
        else:
            LOGGER.info("Skipped feature scaling (scaler=%s).", scaler_name)

    df = df.sort_index()

    return df, label_column, scaler_used, scaled_columns


def apply_loaded_scaler(
    df: pd.DataFrame,
    label_column: Optional[str],
    scaler: Any,
    scaler_columns: List[str],
) -> Tuple[pd.DataFrame, Any, List[str]]:
    """Apply a previously fitted scaler to numeric features."""

    # Verify all scaler columns are present
    missing_cols = set(scaler_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Scaler expects columns that are missing from data: {missing_cols}"
        )

    df = df.copy()

    # Apply scaler only to the specified columns in the correct order
    df.loc[:, scaler_columns] = scaler.transform(df[scaler_columns])

    return df, scaler, scaler_columns


def multi_hot_encode(
    df: pd.DataFrame,
    label_column: Optional[str],
    prefix_columns: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform multi-hot encoding on categorical features."""

    categorical_cols = [
        col
        for col in df.select_dtypes(include=["object"]).columns
        if label_column is None or col != label_column
    ]

    if not categorical_cols:
        return df, pd.DataFrame(index=df.index), []

    categorical_df = df[categorical_cols].copy()
    encoded_frames: List[pd.DataFrame] = []
    processed_columns: List[str] = []

    for column in categorical_df.columns:
        series = categorical_df[column].fillna("")
        non_empty = series[series != ""]
        if non_empty.empty:
            continue

        vectorizer = CountVectorizer(
            tokenizer=lambda value: value.split(","),
            token_pattern=None,
            binary=True,
        )
        vectorizer.fit(non_empty)
        encoded = vectorizer.transform(series)
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=df.index,
        ).astype(bool)

        if prefix_columns:
            prefix = column.split("_")[-1]
            encoded_df.columns = [f"{prefix}_{name}" for name in encoded_df.columns]

        encoded_frames.append(encoded_df)
        processed_columns.append(column)

    if not encoded_frames:
        return df.drop(columns=categorical_df.columns), pd.DataFrame(index=df.index), []

    encoded_df = pd.concat(encoded_frames, axis=1)

    if encoded_df.columns.duplicated().any():
        LOGGER.warning(
            "Duplicate columns detected after multi-hot encoding; merging duplicates with logical OR."
        )
        encoded_df = encoded_df.T.groupby(level=0).any().T.astype(bool)

    if prefix_columns:
        merged_columns: Dict[str, List[str]] = {}
        for column in encoded_df.columns:
            if "_" not in column:
                continue
            _, col_name = column.split("_", 1)
            merged_columns.setdefault(col_name, []).append(column)

        for col_name, columns in merged_columns.items():
            if len(columns) <= 1:
                continue
            all_equal = encoded_df[columns].nunique(axis=1).eq(1).all()
            if all_equal:
                encoded_df[f"merged_{col_name}"] = encoded_df[columns].iloc[:, 0]
                encoded_df = encoded_df.drop(columns=columns)

    df_without_categorical = df.drop(columns=categorical_df.columns)
    return df_without_categorical, encoded_df, processed_columns


def sanitize_dataframe_columns(
    df: pd.DataFrame, label_column: Optional[str]
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Sanitize column names to contain only alphanumeric characters and underscores."""

    rename_map = {col: sanitize_column_name(col) for col in df.columns}
    sanitized_names = list(rename_map.values())
    if len(set(sanitized_names)) != len(sanitized_names):
        duplicates = {
            name for name in sanitized_names if sanitized_names.count(name) > 1
        }
        raise ValueError(f"Column name collision after sanitisation: {duplicates}")

    df = df.rename(columns=rename_map)

    if label_column is None:
        return df, None

    new_label_column = rename_map.get(label_column, label_column)
    if new_label_column != label_column:
        LOGGER.info(
            "Renamed label column from '%s' to '%s' during sanitisation.",
            label_column,
            new_label_column,
        )
    return df, new_label_column


def sanitize_column_name(column: str) -> str:
    """Sanitize a single column name."""

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", column)
    sanitized = re.sub(r"__+", "_", sanitized).strip("_")
    return sanitized or "feature"


def lowercase_and_sort_features(
    df: pd.DataFrame, label_column: Optional[str]
) -> pd.DataFrame:
    """Lowercase feature names and sort them alphabetically."""

    feature_cols = [
        col for col in df.columns if label_column is None or col != label_column
    ]
    lower_map = {col: col.lower() for col in feature_cols}
    lower_names = list(lower_map.values())
    if len(lower_names) != len(set(lower_names)):
        duplicates = {name for name in lower_names if lower_names.count(name) > 1}
        raise ValueError(
            f"Lower-casing feature names introduced duplicates: {duplicates}"
        )

    df = df.rename(columns=lower_map)

    if label_column and label_column in df.columns:
        ordered_columns = [label_column] + sorted(lower_names)
    else:
        ordered_columns = sorted(lower_names)

    return df[ordered_columns]


def scale_numeric_features(
    df: pd.DataFrame,
    label_column: Optional[str],
    scaler_name: str,
) -> Tuple[pd.DataFrame, Optional[StandardScaler], List[str]]:
    """Scale numeric features using the configured scaler."""

    scaler_key = scaler_name.lower()
    if scaler_key not in SCALERS:
        raise ValueError(
            f"Unsupported scaler '{scaler_name}'. Choose from {list(SCALERS)}."
        )

    scaler_cls = SCALERS[scaler_key]
    if scaler_cls is None:
        return df, None, []

    numeric_cols = [
        col
        for col in df.select_dtypes(
            include=["float64", "float32", "int64", "int32", "int16"]
        ).columns
        if label_column is None or col != label_column
    ]

    if not numeric_cols:
        LOGGER.info("No numeric columns found for scaling.")
        return df, None, []

    scaler = scaler_cls()
    df = df.copy()

    # Convert numeric columns to float64 to avoid dtype incompatibility warnings
    for col in numeric_cols:
        df[col] = df[col].astype("float64")

    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler, numeric_cols


def fit_and_apply_scaler_to_splits(
    splits: Dict[str, pd.DataFrame],
    label_column: str,
    scaler_name: str,
) -> Tuple[Optional[Any], List[str]]:
    """Fit scaler on training data only, then transform all splits.

    This prevents data leakage by ensuring validation and test statistics
    don't influence the training preprocessing.

    Args:
        splits: Dictionary of split DataFrames (must contain 'train')
        label_column: Name of the label column to exclude from scaling
        scaler_name: Name of scaler to use ('standard', 'minmax', or 'none')

    Returns:
        Tuple of (fitted_scaler, list_of_scaled_columns)
    """

    scaler_key = scaler_name.lower()
    if scaler_key not in SCALERS:
        raise ValueError(
            f"Unsupported scaler '{scaler_name}'. Choose from {list(SCALERS)}."
        )

    scaler_cls = SCALERS[scaler_key]
    if scaler_cls is None:
        LOGGER.info("Scaling disabled (scaler='none').")
        return None, []

    if "train" not in splits:
        raise ValueError("Training split not found in splits dictionary.")

    train_df = splits["train"]

    # Identify numeric columns (excluding label)
    numeric_cols = [
        col
        for col in train_df.select_dtypes(
            include=["float64", "float32", "int64", "int32", "int16"]
        ).columns
        if col != label_column
    ]

    if not numeric_cols:
        LOGGER.info("No numeric columns found for scaling.")
        return None, []

    # Fit scaler ONLY on training data
    scaler = scaler_cls()
    scaler.fit(train_df[numeric_cols])

    LOGGER.info(
        "Fitted %s scaler on training data: %d features",
        scaler_name,
        len(numeric_cols),
    )

    # Transform all splits with the training-fitted scaler
    for split_name, split_df in splits.items():
        # Convert numeric columns to float64 to avoid dtype incompatibility warnings
        for col in numeric_cols:
            splits[split_name][col] = splits[split_name][col].astype("float64")

        splits[split_name].loc[:, numeric_cols] = scaler.transform(
            split_df[numeric_cols]
        )
        LOGGER.info(
            "Applied scaler to %s split: %d rows",
            split_name,
            len(split_df),
        )

    return scaler, numeric_cols


def enforce_schema_from_metadata(
    df: pd.DataFrame,
    metadata_path: Path,
) -> pd.DataFrame:
    """Enforce schema consistency using model metadata file.

    Ensures all expected columns are present and have correct dtypes.
    Missing columns are created with appropriate fill values.
    """

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    expected_schema = metadata.get("inputs", {})
    if not expected_schema:
        LOGGER.warning("No input schema found in metadata file.")
        return df

    df = df.copy()

    # Track missing and extra columns
    current_cols = set(df.columns)
    expected_cols = set(expected_schema.keys())

    missing_cols = expected_cols - current_cols
    extra_cols = current_cols - expected_cols

    if extra_cols:
        LOGGER.warning(
            "Extra columns not in schema (will be removed): %s", sorted(extra_cols)
        )
        df = df.drop(columns=list(extra_cols))

    if missing_cols:
        LOGGER.info("Adding missing columns from schema: %s", sorted(missing_cols))

        for col in missing_cols:
            dtype = expected_schema[col]

            # Apply the same fill strategy as training
            if dtype == "bool":
                # Boolean columns (multi-hot encoded) default to False
                df[col] = False
            elif dtype in ["float64", "float32", "int64", "int32", "int16"]:
                # Numeric columns default to 0 (same as training NA fill strategy)
                df[col] = 0
            else:
                # Other types default to empty string
                df[col] = ""

            LOGGER.debug(
                "Created column '%s' with dtype=%s fill_value=%s",
                col,
                dtype,
                "False" if dtype == "bool" else "0",
            )

    # Ensure columns are in the same order as metadata
    df = df[list(expected_schema.keys())]

    # Convert dtypes to match metadata
    for col, expected_dtype in expected_schema.items():
        current_dtype = str(df[col].dtype)

        if expected_dtype == "bool" and current_dtype != "bool":
            df[col] = df[col].astype(bool)
        elif expected_dtype.startswith("float") and not current_dtype.startswith(
            "float"
        ):
            df[col] = df[col].astype("float64")
        elif expected_dtype.startswith("int") and not current_dtype.startswith("int"):
            df[col] = df[col].astype("int64")

    LOGGER.info(
        "Schema enforcement complete: expected_cols=%d current_cols=%d missing_added=%d extra_removed=%d",
        len(expected_cols),
        len(df.columns),
        len(missing_cols),
        len(extra_cols),
    )

    return df


def perform_splits(
    df: pd.DataFrame,
    *,
    label_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> Dict[str, pd.DataFrame]:
    """Split the dataset into train/validation/test sets."""

    X = df.drop(columns=[label_column])
    y = df[label_column]

    stratify_series = y if y.nunique() > 1 else None
    remaining = val_size + test_size

    if remaining == 0:
        LOGGER.warning(
            "Validation and test sizes are zero; returning only the training set."
        )
        split_df = combine_target_features(y, X, label_column)
        log_split_distribution("train", y)
        return {"train": split_df}

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=remaining,
        stratify=stratify_series,
        random_state=random_seed,
    )

    splits: Dict[str, pd.DataFrame] = {
        "train": combine_target_features(y_train, X_train, label_column)
    }
    log_split_distribution("train", y_train)

    if val_size == 0 or test_size == 0:
        remainder_name = "test" if test_size else "val"
        splits[remainder_name] = combine_target_features(
            y_remaining, X_remaining, label_column
        )
        log_split_distribution(remainder_name, y_remaining)
        return splits

    test_fraction = test_size / remaining
    stratify_remaining = y_remaining if y_remaining.nunique() > 1 else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=test_fraction,
        stratify=stratify_remaining,
        random_state=random_seed,
    )

    splits["val"] = combine_target_features(y_val, X_val, label_column)
    splits["test"] = combine_target_features(y_test, X_test, label_column)

    log_split_distribution("val", y_val)
    log_split_distribution("test", y_test)

    return splits


def combine_target_features(
    y: pd.Series, X: pd.DataFrame, label_column: str
) -> pd.DataFrame:
    """Combine target and feature matrices into a single DataFrame."""

    return pd.concat([y, X], axis=1)


def log_split_distribution(split_name: str, target: pd.Series) -> None:
    """Log the size and label distribution for a dataset split."""

    distribution = target.value_counts(normalize=True).round(4).to_dict()
    LOGGER.info(
        "Split summary split=%s rows=%d distribution=%s",
        split_name,
        len(target),
        distribution,
    )


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Persist a DataFrame as a parquet file and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(f"Only parquet output is supported. Invalid path: {path}")

    df.to_parquet(path, index=True)
    LOGGER.info(
        "Saved dataframe path=%s rows=%d cols=%d", path, df.shape[0], df.shape[1]
    )
    return path


def save_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    suffix: str,
) -> Dict[str, Path]:
    """Save dataset splits to disk and return the mapping of split names to paths."""

    paths: Dict[str, Path] = {}
    for split_name, split_df in splits.items():
        filename = f"X_{split_name}_{suffix}.parquet"
        path = output_dir / filename
        split_df.to_parquet(path, index=True)
        LOGGER.info(
            "Saved split split=%s path=%s rows=%d cols=%d",
            split_name,
            path,
            split_df.shape[0],
            split_df.shape[1],
        )
        paths[split_name] = path
    return paths


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the CLI interface."""

    parser = argparse.ArgumentParser(
        description="Clean, transform, and split training data for EBM models.",
    )
    parser.add_argument(
        "--positive-path",
        type=Path,
        help="Path to positive instances dataset (parquet or CSV).",
    )
    parser.add_argument(
        "--negative-path",
        type=Path,
        help="Path to negative instances dataset (parquet or CSV).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to a single dataset containing both classes (parquet or CSV).",
    )
    parser.add_argument(
        "--label-column",
        default="y",
        help="Name of the target column when using a single dataset.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Fraction of data to allocate to the training split (default: 0.7).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of data to allocate to the validation split (default: 0.15).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data to allocate to the test split (default: 0.15).",
    )
    parser.add_argument(
        "--suffix",
        default="redux",
        help="Suffix appended to output split filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where split datasets will be stored.",
    )
    parser.add_argument(
        "--cleaned-output-path",
        type=Path,
        help=(
            "Path for the cleaned (unsplit) dataset. Defaults to <output-dir>/"
            "cleaned_dataset_<suffix>.parquet."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=99,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--scaler",
        choices=list(SCALERS.keys()),
        default="standard",
        help="Scaler to apply to numeric features (default: standard).",
    )
    parser.add_argument(
        "--disable-multi-hot",
        action="store_true",
        help="Disable multi-hot encoding for categorical features.",
    )
    parser.add_argument(
        "--disable-prefix",
        action="store_true",
        help="Disable prefixing multi-hot encoded feature names with the source column suffix.",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--mode",
        choices=["training", "inference", "validation"],
        default="training",
        help="Processing mode: 'training' for model training, 'inference' for predictions, 'validation' for dry-run.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Path to model metadata JSON file (required for inference/validation modes).",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        help="Path to save (training) or load (inference) fitted scaler object.",
    )
    parser.add_argument(
        "--schema-diff-output",
        type=Path,
        help="Path to save schema difference report (inference/validation modes).",
    )

    args = parser.parse_args(argv)

    if args.dataset_path and (args.positive_path or args.negative_path):
        parser.error(
            "Provide either --dataset-path or both --positive-path and --negative-path, not both."
        )

    if not args.dataset_path and not (args.positive_path and args.negative_path):
        parser.error(
            "Provide either --dataset-path or both --positive-path and --negative-path."
        )

    return args


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct a PipelineConfig object from CLI arguments."""

    cleaned_output_path = args.cleaned_output_path
    if cleaned_output_path is not None and cleaned_output_path.suffix == "":
        cleaned_output_path = cleaned_output_path.with_suffix(".parquet")

    if cleaned_output_path is None:
        cleaned_output_path = args.output_dir / f"cleaned_dataset_{args.suffix}.parquet"

    return PipelineConfig(
        positive_path=args.positive_path,
        negative_path=args.negative_path,
        dataset_path=args.dataset_path,
        label_column=args.label_column,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        split_suffix=args.suffix,
        split_output_dir=args.output_dir,
        cleaned_output_path=cleaned_output_path,
        random_seed=args.random_seed,
        scaler=args.scaler,
        enable_multi_hot=not args.disable_multi_hot,
        prefix_multi_hot=not args.disable_prefix,
        mode=args.mode,
        metadata_path=args.metadata_path,
        scaler_path=args.scaler_path,
        schema_diff_output=args.schema_diff_output,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI interface."""

    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        config = build_config_from_args(args)
        result = execute_data_pipeline(config)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Pipeline execution failed: %s", exc)
        raise SystemExit(1) from exc

    if result.schema_diff:
        print("\n" + str(result.schema_diff))

    LOGGER.info(
        "Pipeline completed successfully mode=%s cleaned_path=%s splits=%s",
        config.mode,
        result.cleaned_path,
        {name: str(path) for name, path in result.split_paths.items()},
    )


# ============================================================================
# Legacy API Wrapper Class
# ============================================================================


class FeatureCleaning:
    """
    Wrapper class for backward compatibility with predictor.py.

    Provides a simple class-based interface to the function-based API.
    """

    def __init__(self, mode: str = "inference"):
        """
        Initialize feature cleaner.

        Parameters
        ----------
        mode : str
            Cleaning mode: 'training', 'inference', or 'validation'
        """
        self.mode = mode

    def clean(
        self,
        df: pd.DataFrame,
        metadata_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Clean and transform features for ML inference.

        Parameters
        ----------
        df : pd.DataFrame
            Input features
        metadata_path : str, optional
            Path to model schema JSON file
        scaler_path : str, optional
            Path to saved scaler pickle file

        Returns
        -------
        pd.DataFrame
            Cleaned and transformed features
        """
        # Apply basic cleaning
        df = apply_cleaning_steps(df, label_column=None)
        df = add_length_feature(df)
        df = drop_non_feature_columns(df)
        df = handle_missing_values(df)

        # Load and apply scaler if provided
        if scaler_path:
            scaler, feature_names = load_scaler(Path(scaler_path))
            # Align columns to scaler features
            missing_cols = set(feature_names) - set(df.columns)
            extra_cols = set(df.columns) - set(feature_names)

            if missing_cols:
                LOGGER.warning(f"Adding {len(missing_cols)} missing columns with zeros")
                for col in missing_cols:
                    df[col] = 0

            if extra_cols:
                LOGGER.warning(f"Dropping {len(extra_cols)} extra columns")
                df = df.drop(columns=list(extra_cols))

            # Reorder to match scaler
            df = df[feature_names]

            # Apply scaling
            df_scaled = pd.DataFrame(
                scaler.transform(df), columns=feature_names, index=df.index
            )
            df = df_scaled

        return df


if __name__ == "__main__":
    main()
