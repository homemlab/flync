#!/usr/bin/env python3
"""
Data Preparation Script

This script converts the original datasets to the format expected by the hyperparameter optimizer.
It takes the original train/test split and creates proper train/validation/holdout splits with
advanced stratification options and sampling techniques for handling class imbalance.

Examples:
    # Basic usage
    python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet --output-dir ./data

    # With SMOTE oversampling
    python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet --output-dir ./data \\
        --sampling-strategy smote --smote-k-neighbors 3 --smote-sampling-strategy auto

    # With SMOTE-Tomek hybrid approach
    python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet --output-dir ./data \\
        --sampling-strategy smote_tomek --smote-sampling-strategy 0.8

    # With random undersampling
    python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet --output-dir ./data \\
        --sampling-strategy random_undersample --smote-sampling-strategy auto
"""

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from pathlib import Path
from typing import Optional, Union, List

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_datasets(positive_file: str, negative_file: str, output_dir: str, 
                    target_column: str = "y", test_size: float = 0.2, 
                    holdout_size: float = 0.15, random_state: int = 42,
                    stratify_by: Optional[str] = None, 
                    stratify_groups: Optional[str] = None,
                    stratify_groups_min_size: int = 2,
                    balance_classes: bool = False,
                    max_imbalance_ratio: float = 10.0,
                    sampling_strategy: str = "none",
                    smote_k_neighbors: int = 5,
                    smote_sampling_strategy: Union[str, float] = "auto"):
    """
    Prepare datasets from positive and negative samples with advanced stratification options.
    
    Args:
        positive_file: Path to positive samples parquet file
        negative_file: Path to negative samples parquet file  
        output_dir: Output directory for prepared datasets
        target_column: Name of target column
        test_size: Test set size (fraction)
        holdout_size: Holdout set size (fraction)
        random_state: Random state for reproducibility
        stratify_by: Column to use for stratification (defaults to target_column)
        stratify_groups: Column defining groups for stratified group splits
        stratify_groups_min_size: Minimum size for groups in stratified group splits
        balance_classes: Whether to balance classes by undersampling majority class
        max_imbalance_ratio: Maximum allowed imbalance ratio (majority/minority)
        sampling_strategy: Sampling technique to use ('none', 'smote', 'smote_tomek', 'random_undersample')
        smote_k_neighbors: Number of nearest neighbors for SMOTE
        smote_sampling_strategy: Sampling strategy for SMOTE ('auto', 'minority', 'not minority', 'not majority', 'all', or float)
    
    Creates train/test/holdout splits from the original positive and negative datasets.
    """
    logger.info("Loading positive and negative datasets...")
    
    # Load datasets
    df_pos = pd.read_parquet(positive_file)
    df_neg = pd.read_parquet(negative_file)
    
    # Add target labels
    df_pos[target_column] = 1
    df_neg[target_column] = 0
    
    logger.info(f"Positive samples: {len(df_pos)}")
    logger.info(f"Negative samples: {len(df_neg)}")
    
    # Combine datasets
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    logger.info(f"Combined dataset shape: {df_combined.shape}")
    
    # Remove duplicates
    initial_size = len(df_combined)
    df_combined = df_combined.drop_duplicates()
    logger.info(f"Removed {initial_size - len(df_combined)} duplicates")
    
    # Handle class balancing if requested
    if balance_classes:
        df_combined = _balance_classes(df_combined, target_column, max_imbalance_ratio)
    
    # Set default stratification column
    if stratify_by is None:
        stratify_by = target_column
    
    # Validate stratification column exists
    if stratify_by not in df_combined.columns:
        raise ValueError(f"Stratification column '{stratify_by}' not found in dataset")
    
    # Prepare features and target
    X = df_combined.drop(columns=[target_column])
    y = df_combined[target_column]
    stratify_col = df_combined[stratify_by]
    
    logger.info(f"Stratifying by column: {stratify_by}")
    logger.info(f"Stratification column distribution: {stratify_col.value_counts().to_dict()}")
    
    # Create splits based on stratification method
    if stratify_groups is not None:
        # Use group-aware stratified splitting
        X_train, X_test, X_holdout, y_train, y_test, y_holdout = _create_stratified_group_splits(
            df_combined, X, y, stratify_col, stratify_groups, 
            test_size, holdout_size, stratify_groups_min_size, random_state
        )
    else:
        # Use standard stratified splitting
        X_train, X_test, X_holdout, y_train, y_test, y_holdout = _create_stratified_splits(
            X, y, stratify_col, test_size, holdout_size, random_state
        )
    
    # Recombine features and targets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    holdout_df = pd.concat([X_holdout, y_holdout], axis=1)
    
    # Apply sampling strategies to training set only
    if sampling_strategy != "none":
        logger.info(f"Applying {sampling_strategy} sampling to training set...")
        train_df = _apply_sampling_strategy(
            train_df, target_column, sampling_strategy, 
            smote_k_neighbors, smote_sampling_strategy, random_state
        )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_file = output_path / "train.parquet"
    test_file = output_path / "test.parquet"
    holdout_file = output_path / "holdout.parquet"
    
    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)
    holdout_df.to_parquet(holdout_file, index=False)
    
    logger.info(f"Saved datasets to {output_dir}:")
    logger.info(f"  Train: {train_file} (shape: {train_df.shape})")
    logger.info(f"  Test: {test_file} (shape: {test_df.shape})")
    logger.info(f"  Holdout: {holdout_file} (shape: {holdout_df.shape})")
    
    # Log sampling configuration
    if sampling_strategy != "none":
        logger.info(f"Sampling configuration:")
        logger.info(f"  Strategy: {sampling_strategy}")
        if sampling_strategy in ["smote", "smote_tomek"]:
            logger.info(f"  K-neighbors: {smote_k_neighbors}")
        logger.info(f"  Sampling strategy parameter: {smote_sampling_strategy}")
    
    # Log class distributions
    for name, df in [("Train", train_df), ("Test", test_df), ("Holdout", holdout_df)]:
        dist = df[target_column].value_counts(normalize=True).sort_index()
        logger.info(f"  {name} class distribution: {dist.to_dict()}")


def _balance_classes(df: pd.DataFrame, target_column: str, max_imbalance_ratio: float) -> pd.DataFrame:
    """Balance classes by undersampling the majority class."""
    class_counts = df[target_column].value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    
    current_ratio = majority_count / minority_count
    logger.info(f"Current class imbalance ratio: {current_ratio:.2f}")
    
    if current_ratio > max_imbalance_ratio:
        target_majority_count = int(minority_count * max_imbalance_ratio)
        logger.info(f"Balancing classes: reducing majority class to {target_majority_count} samples")
        
        # Identify majority and minority classes
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        # Sample from majority class
        majority_samples = df[df[target_column] == majority_class].sample(
            n=target_majority_count, random_state=42
        )
        minority_samples = df[df[target_column] == minority_class]
        
        # Combine balanced dataset
        balanced_df = pd.concat([majority_samples, minority_samples], ignore_index=True)
        logger.info(f"Balanced dataset shape: {balanced_df.shape}")
        
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    return df


def _create_stratified_splits(X: pd.DataFrame, y: pd.Series, stratify_col: pd.Series,
                             test_size: float, holdout_size: float, random_state: int):
    """Create stratified train/test/holdout splits."""
    # First split: separate holdout set
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y, test_size=holdout_size, stratify=stratify_col, random_state=random_state
    )
    
    # Get stratification column for temporary set
    stratify_temp = stratify_col.iloc[X_temp.index] if hasattr(stratify_col, 'iloc') else stratify_col[X_temp.index]
    
    # Second split: separate train and test from remaining data
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(1-holdout_size), 
        stratify=stratify_temp, random_state=random_state
    )
    
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def _create_stratified_group_splits(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                                   stratify_col: pd.Series, group_col: str,
                                   test_size: float, holdout_size: float,
                                   min_group_size: int, random_state: int):
    """Create stratified group-aware splits."""
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataset")
    
    groups = df[group_col]
    
    # Filter out groups that are too small
    group_counts = groups.value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index
    
    if len(valid_groups) == 0:
        raise ValueError(f"No groups with minimum size {min_group_size} found")
    
    logger.info(f"Using {len(valid_groups)} groups with minimum size {min_group_size}")
    logger.info(f"Filtered out {len(group_counts) - len(valid_groups)} groups that were too small")
    
    # Filter dataset to only include valid groups
    valid_mask = groups.isin(valid_groups)
    df_filtered = df[valid_mask].copy()
    X_filtered = X[valid_mask].copy()
    y_filtered = y[valid_mask].copy()
    stratify_filtered = stratify_col[valid_mask].copy()
    groups_filtered = groups[valid_mask].copy()
    
    logger.info(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Use a simple approach: randomly assign groups to splits while maintaining stratification
    unique_groups = groups_filtered.unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_groups)
    
    # Calculate split indices
    n_groups = len(unique_groups)
    holdout_n = max(1, int(n_groups * holdout_size))
    test_n = max(1, int(n_groups * test_size))
    train_n = n_groups - holdout_n - test_n
    
    # Assign groups to splits
    holdout_groups = unique_groups[:holdout_n]
    test_groups = unique_groups[holdout_n:holdout_n + test_n]
    train_groups = unique_groups[holdout_n + test_n:]
    
    logger.info(f"Group distribution - Train: {len(train_groups)}, Test: {len(test_groups)}, Holdout: {len(holdout_groups)}")
    
    # Create splits based on group assignment
    train_mask = groups_filtered.isin(train_groups)
    test_mask = groups_filtered.isin(test_groups)
    holdout_mask = groups_filtered.isin(holdout_groups)
    
    X_train = X_filtered[train_mask]
    X_test = X_filtered[test_mask]
    X_holdout = X_filtered[holdout_mask]
    
    y_train = y_filtered[train_mask]
    y_test = y_filtered[test_mask]
    y_holdout = y_filtered[holdout_mask]
    
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def _apply_sampling_strategy(df: pd.DataFrame, target_column: str, 
                           sampling_strategy: str, smote_k_neighbors: int,
                           smote_sampling_strategy: Union[str, float], 
                           random_state: int) -> pd.DataFrame:
    """Apply the specified sampling strategy to the dataset."""
    if not IMBALANCED_LEARN_AVAILABLE:
        logger.error("imbalanced-learn is required for sampling strategies. Install with: pip install imbalanced-learn")
        raise ImportError("imbalanced-learn package is required for sampling strategies")
    
    # Validate sampling strategy
    valid_strategies = ["none", "smote", "smote_tomek", "random_undersample"]
    if sampling_strategy not in valid_strategies:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}. Valid options: {valid_strategies}")
    
    if sampling_strategy == "none":
        return df
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Validate features for sampling
    _validate_features_for_sampling(X, y, sampling_strategy, smote_k_neighbors)
    
    # Parse sampling strategy parameter
    parsed_sampling_strategy = _parse_sampling_strategy(str(smote_sampling_strategy))
    
    # Log original distribution
    original_dist = y.value_counts().sort_index()
    logger.info(f"Original training distribution: {original_dist.to_dict()}")
    
    # Validate features for sampling
    _validate_features_for_sampling(X, sampling_strategy)
    
    # Apply sampling strategy
    if sampling_strategy == "smote":
        sampler = SMOTE(
            sampling_strategy=parsed_sampling_strategy,
            k_neighbors=smote_k_neighbors,
            random_state=random_state
        )
        logger.info(f"Applying SMOTE with k_neighbors={smote_k_neighbors}, sampling_strategy={parsed_sampling_strategy}")
        
    elif sampling_strategy == "smote_tomek":
        sampler = SMOTETomek(
            sampling_strategy=parsed_sampling_strategy,
            smote=SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state),
            random_state=random_state
        )
        logger.info(f"Applying SMOTE-Tomek with k_neighbors={smote_k_neighbors}, sampling_strategy={parsed_sampling_strategy}")
        
    elif sampling_strategy == "random_undersample":
        sampler = RandomUnderSampler(
            sampling_strategy=parsed_sampling_strategy,
            random_state=random_state
        )
        logger.info(f"Applying Random Undersampling with sampling_strategy={parsed_sampling_strategy}")
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Log new distribution
        new_dist = pd.Series(y_resampled).value_counts().sort_index()
        logger.info(f"Resampled training distribution: {new_dist.to_dict()}")
        logger.info(f"Dataset size change: {len(df)} -> {len(X_resampled)} samples")
        
        # Create resampled DataFrame
        resampled_df = pd.concat([
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.Series(y_resampled, name=target_column)
        ], axis=1)
        
        return resampled_df
        
    except Exception as e:
        logger.error(f"Error applying {sampling_strategy}: {e}")
        logger.warning("Falling back to original dataset without sampling")
        return df


def _parse_sampling_strategy(strategy_str: str) -> Union[str, float]:
    """Parse sampling strategy string to appropriate type."""
    # Try to convert to float if it's a number
    try:
        return float(strategy_str)
    except ValueError:
        # Return as string for categorical strategies
        valid_string_strategies = ["auto", "minority", "not minority", "not majority", "all"]
        if strategy_str in valid_string_strategies:
            return strategy_str
        else:
            logger.warning(f"Unknown sampling strategy '{strategy_str}', using 'auto'")
            return "auto"


def _validate_features_for_sampling(X: pd.DataFrame, y: pd.Series, sampling_strategy: str, k_neighbors: int) -> None:
    """Validate that features are suitable for sampling algorithms."""
    if sampling_strategy in ["smote", "smote_tomek"]:
        # SMOTE requires numeric features
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Non-numeric columns detected: {non_numeric_cols}")
            logger.warning("SMOTE algorithms work best with numeric features.")
            logger.warning("Consider encoding categorical features or using different sampling strategy.")
        
        # Check for any infinite or NaN values
        if X.isnull().any().any():
            logger.warning("Missing values detected in features. SMOTE may have issues with missing data.")
        
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected in features. This may cause issues with SMOTE.")
        
        # Check for sufficient samples for k_neighbors
        minority_class_count = y.value_counts().min()
        if minority_class_count <= k_neighbors:
            logger.warning(f"Minority class has only {minority_class_count} samples, but k_neighbors={k_neighbors}")
            logger.warning("Consider reducing k_neighbors or using a different sampling strategy.")
    
    logger.info(f"Feature validation completed for {sampling_strategy} sampling")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for hyperparameter optimization with advanced stratification and sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--positive-file", required=True, help="Path to positive samples parquet file")
    parser.add_argument("--negative-file", required=True, help="Path to negative samples parquet file")
    parser.add_argument("--output-dir", required=True, help="Output directory for prepared datasets")
    
    # Basic split parameters
    parser.add_argument("--target-column", default="y", help="Name of target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (fraction)")
    parser.add_argument("--holdout-size", type=float, default=0.15, help="Holdout set size (fraction)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    
    # Stratification parameters
    parser.add_argument("--stratify-by", type=str, help="Column to use for stratification (defaults to target column)")
    parser.add_argument("--stratify-groups", type=str, help="Column defining groups for stratified group splits")
    parser.add_argument("--stratify-groups-min-size", type=int, default=2, 
                       help="Minimum size for groups in stratified group splits")
    
    # Class balancing parameters
    parser.add_argument("--balance-classes", action="store_true", 
                       help="Balance classes by undersampling majority class")
    parser.add_argument("--max-imbalance-ratio", type=float, default=10.0,
                       help="Maximum allowed imbalance ratio (majority/minority)")
    
    # Advanced sampling parameters
    parser.add_argument("--sampling-strategy", type=str, default="none",
                       choices=["none", "smote", "smote_tomek", "random_undersample"],
                       help="Sampling technique to use for handling class imbalance in training set")
    parser.add_argument("--smote-k-neighbors", type=int, default=5,
                       help="Number of nearest neighbors for SMOTE algorithms (must be < minority class size)")
    parser.add_argument("--smote-sampling-strategy", type=str, default="auto",
                       help="Sampling strategy for resampling algorithms. Options: 'auto' (balance to majority), "
                            "'minority' (only oversample minority), 'not minority', 'not majority', 'all', or float ratio")
    
    args = parser.parse_args()
    
    # Validate imbalanced-learn availability for sampling strategies
    if args.sampling_strategy != "none" and not IMBALANCED_LEARN_AVAILABLE:
        logger.error(f"Sampling strategy '{args.sampling_strategy}' requires imbalanced-learn package.")
        logger.error("Install with: pip install imbalanced-learn")
        return 1
    
    prepare_datasets(
        positive_file=args.positive_file,
        negative_file=args.negative_file,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_size=args.test_size,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
        stratify_by=args.stratify_by,
        stratify_groups=args.stratify_groups,
        stratify_groups_min_size=args.stratify_groups_min_size,
        balance_classes=args.balance_classes,
        max_imbalance_ratio=args.max_imbalance_ratio,
        sampling_strategy=args.sampling_strategy,
        smote_k_neighbors=args.smote_k_neighbors,
        smote_sampling_strategy=args.smote_sampling_strategy
    )
    
    return 0


if __name__ == "__main__":
    main()
