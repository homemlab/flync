#!/usr/bin/env python3
"""
Hyperparameter Optimization CLI Tool

A command-line tool for hyperparameter optimization of ML models using Optuna and MLflow.
Supports RandomForest, XGBoost, LightGBM, and EBM classifiers with multiple optimization metrics.

Features:
- Multi-metric optimization with Optuna pruning for efficiency
- Comprehensive MLflow tracking and artifact logging
- Feature importance analysis and stability plots
- Feature selection via correlation analysis or explicit lists
- EBM explainability integration with visualization support
- Robust error handling and fallback mechanisms
- Both CLI and Python API interfaces

Performance Notes:
- Uses TPESampler and MedianPruner for efficient hyperparameter search
- Supports parallel training via threads parameter
- For large-scale optimization, consider distributed Optuna with appropriate storage backends
"""

import argparse
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import tempfile

    import lightgbm as lgb
    import mlflow
    import mlflow.data
    import mlflow.lightgbm
    import mlflow.models
    import mlflow.pyfunc
    import mlflow.sklearn
    import mlflow.tracking
    import mlflow.xgboost
    import numpy as np
    import optuna
    import pandas as pd
    import xgboost as xgb
    from interpret.glassbox import ExplainableBoostingClassifier
    from mlflow.data.pandas_dataset import PandasDataset
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    # Optional imports with graceful degradation
    try:
        import joblib
    except ImportError:
        joblib = None
        print(
            "Warning: joblib not available - RandomForest model saving may be limited",
            file=sys.stderr,
        )

except ImportError as e:
    error_msg = f"Missing required dependency: {e}\nPlease install requirements: pip install -r requirements.txt"
    logger.info(error_msg, file=sys.stderr)
    raise RuntimeError(error_msg) from e

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hyperparameter_optimization.log"),
    ],
)
logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Lightweight .env loader to populate environment variables.

    Searches upward from this file location and the current working directory
    for a file named '.env' and exports KEY=VALUE pairs into os.environ if not set.
    Comments and blank lines are ignored. Quotes around values are stripped.
    """
    try:
        candidates: List[Path] = []
        # CWD first
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            candidates.append(cwd_env)

        # Walk up from this file's directory
        for parent in Path(__file__).resolve().parents:
            env_path = parent / ".env"
            if env_path.exists():
                candidates.append(env_path)
                break

        for env_file in candidates:
            try:
                with env_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                logger.debug(f"Loaded environment variables from {env_file}")
                # stop after first successfully loaded .env
                break
            except Exception as e:
                logger.debug(f"Failed loading .env from {env_file}: {e}")
    except Exception as e:
        logger.debug(f".env loading skipped due to error: {e}")


class FeatureAnalyzer:
    """
    A class to analyze features, compute a correlation matrix, identify highly
    correlated features, and suggest a list of features to drop.
    """

    @staticmethod
    def analyze_and_log_correlation(
        X_train: pd.DataFrame, threshold: float, output_dir: Path = None
    ) -> List[str]:
        """
        Calculates the correlation matrix, logs it directly to MLflow, and returns features to drop.
        """
        logger.info(f"Starting feature correlation analysis with threshold > {threshold}...")
        corr_matrix = X_train.corr().abs()

        # Plotting the heatmap
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(corr_matrix, cmap="viridis", annot=False, ax=ax)
        ax.set_title("Feature Correlation Matrix")

        # Log figure directly to MLflow (more reliable than file-based logging)
        mlflow.log_figure(fig, "feature_analysis/feature_correlation_matrix.png")
        plt.close(fig)
        logger.info("Correlation matrix plot logged to MLflow.")

        # Identify and get list of highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features_to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]

        logger.info(f"Identified {len(features_to_drop)} features to drop.")

        # Create the list of features to drop as text and log it directly to MLflow
        features_text = "\n".join(features_to_drop) if features_to_drop else "No features to drop"
        mlflow.log_text(features_text, "feature_analysis/dropped_features_correlated.txt")
        logger.info("List of features to drop logged to MLflow.")

        return features_to_drop


class DataLoader:
    """Handles loading and validation of datasets."""

    @staticmethod
    def load_datasets(
        train_path: str, test_path: str, holdout_path: str
    ) -> Tuple[pd.DataFrame, ...]:
        """Load train, test, and holdout datasets."""
        logger.info("Loading datasets...")

        try:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            holdout_df = pd.read_parquet(holdout_path)

            logger.info(f"Train dataset shape: {train_df.shape}")
            logger.info(f"Test dataset shape: {test_df.shape}")
            logger.info(f"Holdout dataset shape: {holdout_df.shape}")

            return train_df, test_df, holdout_df

        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise

    @staticmethod
    def create_mlflow_datasets(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        train_path: str,
        test_path: str,
        holdout_path: str,
        target_column: str,
        dataset_version: str = None,
        dataset_suffix: str = None,
    ) -> Tuple[PandasDataset, ...]:
        """Create MLflow Dataset objects for proper dataset tracking."""

        # Create dataset metadata
        base_name = dataset_suffix or "training_dataset"
        version_number = dataset_version or "v1.0"

        logger.debug(f"Creating MLflow datasets with base name: {base_name}")
        logger.debug(f"Using dataset version number: {version_number}")
        logger.debug(f"Using target column: {target_column}")

        # Validate target column exists in datasets
        if target_column not in train_df.columns:
            logger.warning(
                f"Target column '{target_column}' not found in train dataset. Available columns: {list(train_df.columns)}"
            )
            target_column = None

        try:
            # Try the newer MLflow API with version parameter
            train_dataset = mlflow.data.from_pandas(
                train_df,
                source=train_path,
                name=f"{base_name}_train",
                version=version_number,
                targets=target_column,
            )

            test_dataset = mlflow.data.from_pandas(
                test_df,
                source=test_path,
                name=f"{base_name}_test",
                version=version_number,
                targets=target_column,
            )

            holdout_dataset = mlflow.data.from_pandas(
                holdout_df,
                source=holdout_path,
                name=f"{base_name}_holdout",
                version=version_number,
                targets=target_column,
            )

            logger.info(f"Created MLflow datasets with version number: {version_number}")

        except TypeError as e:
            if "version" in str(e):
                logger.warning(f"MLflow version doesn't support 'version' parameter: {e}")
                logger.info("Falling back to basic dataset creation without version parameter")

                # Fallback to older MLflow API without version parameter
                train_dataset = mlflow.data.from_pandas(
                    train_df,
                    source=train_path,
                    name=f"{base_name}_train",
                    targets=target_column,
                )

                test_dataset = mlflow.data.from_pandas(
                    test_df,
                    source=test_path,
                    name=f"{base_name}_test",
                    targets=target_column,
                )

                holdout_dataset = mlflow.data.from_pandas(
                    holdout_df,
                    source=holdout_path,
                    name=f"{base_name}_holdout",
                    targets=target_column,
                )

                logger.info(
                    f"Created MLflow datasets (without version) - dataset_version will be logged as tag: {version_number}"
                )
            else:
                raise e

        except Exception as e:
            logger.error(f"Error creating MLflow datasets: {e}")
            raise

        return train_dataset, test_dataset, holdout_dataset

    @staticmethod
    def prepare_features_and_targets(
        df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Start with all columns except target
        X = df.drop(columns=[target_column])

        # Identify columns with 'object' dtype, which are not suitable for most models
        object_columns = X.select_dtypes(include=["object"]).columns

        if len(object_columns) > 0:
            error_message = (
                f"Dataset contains columns with 'object' dtype, which is not supported: "
                f"{list(object_columns)}. Please ensure all feature columns are numeric "
                f"(e.g., int, float, bool) or properly encoded before optimization."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        # Identify other non-numeric columns (like 'category') and log a warning
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_columns) > 0:
            logger.warning(
                f"Found non-numeric columns that are not of 'object' type: {list(non_numeric_columns)}. "
                "These will be passed to the model but may require encoding for optimal performance."
            )

        y = df[target_column]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    @staticmethod
    def apply_feature_selection(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        target_column: str,
        analyze_correlations: bool = False,
        correlation_threshold: float = 0.95,
        drop_features_file: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Apply feature selection based on correlation analysis and/or explicit feature list."""
        features_to_drop = []

        # Analyze correlations if requested
        if analyze_correlations:
            X_train, _ = DataLoader.prepare_features_and_targets(train_df, target_column)
            corr_features = FeatureAnalyzer.analyze_and_log_correlation(
                X_train, correlation_threshold
            )
            features_to_drop.extend(corr_features)
            logger.info(f"Added {len(corr_features)} highly correlated features to drop list")

        # Load features from file if provided
        if drop_features_file:
            if not os.path.exists(drop_features_file):
                raise FileNotFoundError(f"Features file not found: {drop_features_file}")

            with open(drop_features_file, "r") as f:
                file_features = [line.strip() for line in f if line.strip()]
            features_to_drop.extend(file_features)
            logger.info(f"Added {len(file_features)} features from file to drop list")

        # Remove duplicates while preserving order
        features_to_drop = list(dict.fromkeys(features_to_drop))

        if features_to_drop:
            # Verify features exist in datasets before dropping
            available_features = set(train_df.columns) - {target_column}
            valid_features_to_drop = [f for f in features_to_drop if f in available_features]
            invalid_features = [f for f in features_to_drop if f not in available_features]

            if invalid_features:
                logger.warning(f"Features not found in dataset, skipping: {invalid_features}")

            if valid_features_to_drop:
                logger.info(
                    f"Dropping {len(valid_features_to_drop)} features: {valid_features_to_drop[:10]}{'...' if len(valid_features_to_drop) > 10 else ''}"
                )
                train_df = train_df.drop(columns=valid_features_to_drop)
                test_df = test_df.drop(columns=valid_features_to_drop)
                holdout_df = holdout_df.drop(columns=valid_features_to_drop)

                # Log dropped features to MLflow
                mlflow.log_text(
                    "\n".join(valid_features_to_drop),
                    "feature_selection/dropped_features.txt",
                )
                mlflow.log_param("num_features_dropped", len(valid_features_to_drop))
                mlflow.log_param("original_feature_count", len(available_features))
                mlflow.log_param(
                    "final_feature_count",
                    len(available_features) - len(valid_features_to_drop),
                )

        return train_df, test_df, holdout_df

    @staticmethod
    def validate_file_paths(*paths: str) -> None:
        """Validate that all file paths exist."""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")


class MetricsCalculator:
    """Handles calculation and logging of various metrics, including confusion matrix and CSVs."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for binary classification."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics["roc_auc"] = 0.0
            logger.warning("ROC AUC could not be calculated (single class present)")

        # Calculate PR AUC
        try:
            precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics["pr_auc"] = auc(recalls, precisions)
        except ValueError:
            metrics["pr_auc"] = 0.0
            logger.warning("PR AUC could not be calculated (single class present)")

        return metrics

    @staticmethod
    def log_curves_and_confusion(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        prefix: str,
        trial_number: Optional[int] = None,
    ) -> None:
        """Log PR, ROC curves, confusion matrix (plots and CSVs) to MLflow."""
        import pandas as pd

        suffix = f"_trial_{trial_number}" if trial_number is not None else ""

        # PR Curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recalls, precisions)
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        ax_pr.plot(recalls, precisions, label=f"PR Curve (AUC = {pr_auc:.3f})")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{prefix} PR Curve{suffix}")
        ax_pr.legend()
        ax_pr.grid(True, alpha=0.3)
        mlflow.log_figure(fig_pr, f"plots/{prefix}_pr_curve{suffix}.png")
        plt.close(fig_pr)
        # Save PR curve data as CSV
        pr_df = pd.DataFrame({"recall": recalls, "precision": precisions})
        mlflow.log_text(pr_df.to_csv(index=False), f"plots/{prefix}_pr_curve{suffix}.csv")

        # ROC Curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
            ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"{prefix} ROC Curve{suffix}")
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)
            mlflow.log_figure(fig_roc, f"plots/{prefix}_roc_curve{suffix}.png")
            plt.close(fig_roc)
            # Save ROC curve data as CSV
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            mlflow.log_text(roc_df.to_csv(index=False), f"plots/{prefix}_roc_curve{suffix}.csv")
        except ValueError:
            logger.warning("ROC curve could not be generated")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax_cm.figure.colorbar(im, ax=ax_cm)
        ax_cm.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=[0, 1],
            yticklabels=[0, 1],
            ylabel="True label",
            xlabel="Predicted label",
            title=f"{prefix} Confusion Matrix{suffix}",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        mlflow.log_figure(fig_cm, f"plots/{prefix}_confusion_matrix{suffix}.png")
        plt.close(fig_cm)
        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["True_0", "True_1"])
        mlflow.log_text(cm_df.to_csv(), f"plots/{prefix}_confusion_matrix{suffix}.csv")


class ModelOptimizer:
    """Handles model creation and hyperparameter optimization, including EBM support."""

    def __init__(
        self,
        model_type: str,
        optimization_metrics: List[str],
        optimization_direction: str,
        random_state: int = 42,
        threads: int = 1,
        light_artifacts: bool = False,
        trial_timeout: int = 7200,
    ):
        self.model_type = model_type.lower()
        self.optimization_metrics = optimization_metrics
        self.optimization_direction = optimization_direction
        self.random_state = random_state
        self.threads = threads
        self.light_artifacts = light_artifacts
        self.trial_timeout = trial_timeout
        self._feature_count: Optional[int] = None  # For adaptive parameter selection

        if self.model_type not in ["randomforest", "xgboost", "ebm", "lightgbm"]:
            raise ValueError("Model type must be 'randomforest', 'xgboost', 'ebm', or 'lightgbm'")

        valid_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        for metric in optimization_metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics: {valid_metrics}")

    def get_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space based on model type."""
        if self.model_type == "randomforest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_float("min_samples_split", 0.01, 1.0),
                "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.01, 0.5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced", "balanced_subsample"]
                ),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "random_state": self.random_state,
                "n_jobs": self.threads,
            }
        elif self.model_type == "xgboost":
            return {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
                "random_state": self.random_state,
                "n_jobs": self.threads,
                "verbosity": 0,
            }
        elif self.model_type == "ebm":
            # Detect large dataset and adjust parameters accordingly
            is_large_dataset = (hasattr(self, '_feature_count') and 
                              self._feature_count is not None and 
                              self._feature_count > 5000)
            
            # Conservative parameters for large datasets (8000+ features)
            if is_large_dataset:
                return {
                    # Core binning parameters - heavily constrained for large datasets
                    "max_bins": trial.suggest_int("max_bins", 32, 64),  # Reduced from 256
                    "max_interaction_bins": trial.suggest_int("max_interaction_bins", 8, 16),  # Reduced
                    # Interaction configuration - disabled for large datasets
                    "interactions": trial.suggest_categorical("interactions", [0, 1]),  # Max 1 interaction
                    # Learning and boosting parameters
                    "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                    "greedy_ratio": trial.suggest_float("greedy_ratio", 0.0, 5.0),  # Reduced range
                    "cyclic_progress": trial.suggest_float("cyclic_progress", 0.9, 1.0),
                    # Bagging parameters - minimized for memory efficiency
                    "outer_bags": trial.suggest_int("outer_bags", 4, 8),  # Reduced
                    "inner_bags": trial.suggest_int("inner_bags", 0, 1),  # Minimal
                    # Regularization parameters
                    "validation_size": trial.suggest_float("validation_size", 0.15, 0.25),
                    "smoothing_rounds": trial.suggest_int("smoothing_rounds", 0, 20),  # Reduced
                    "interaction_smoothing_rounds": trial.suggest_int(
                        "interaction_smoothing_rounds", 5, 25  # Heavily reduced
                    ),
                    # Tree structure parameters
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),  # Higher for stability
                    "max_leaves": trial.suggest_int("max_leaves", 3, 5),  # Reduced complexity
                    "min_hessian": trial.suggest_float("min_hessian", 1e-3, 1e-1, log=True),
                    # L1/L2 regularization - increased for large datasets
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
                    # Early stopping parameters - much more aggressive for large datasets
                    "max_rounds": trial.suggest_int("max_rounds", 100, 500),  # Drastically reduced
                    "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 25),  # Earlier stopping
                    "early_stopping_tolerance": trial.suggest_float(
                        "early_stopping_tolerance", 1e-4, 1e-2, log=True
                    ),
                    # Categorical handling parameters
                    "gain_scale": trial.suggest_float("gain_scale", 1.0, 3.0),
                    "min_cat_samples": trial.suggest_int("min_cat_samples", 20, 50),  # Higher threshold
                    "cat_smooth": trial.suggest_float("cat_smooth", 10.0, 30.0),
                    # Missing value handling
                    "missing": trial.suggest_categorical("missing", ["separate"]),  # Most stable option
                    # Fixed parameters
                    "n_jobs": min(self.threads, 4),  # Cap threads for large datasets
                    "random_state": self.random_state,
                }
            else:
                # Original parameters for smaller datasets
                return {
                    # Core binning parameters - constrained for large datasets
                    "max_bins": trial.suggest_int("max_bins", 64, 256),
                    "max_interaction_bins": trial.suggest_int("max_interaction_bins", 16, 64),
                    # Interaction configuration - safe numeric values only
                    "interactions": trial.suggest_categorical("interactions", [0, 1, 2, 3]),
                    # Learning and boosting parameters
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "greedy_ratio": trial.suggest_float("greedy_ratio", 0.0, 20.0),
                    "cyclic_progress": trial.suggest_float("cyclic_progress", 0.8, 1.0),
                    # Bagging parameters - reduced for faster training
                    "outer_bags": trial.suggest_int("outer_bags", 8, 16),
                    "inner_bags": trial.suggest_int("inner_bags", 0, 3),
                    # Regularization parameters
                    "validation_size": trial.suggest_float("validation_size", 0.15, 0.25),
                    "smoothing_rounds": trial.suggest_int("smoothing_rounds", 0, 50),
                    "interaction_smoothing_rounds": trial.suggest_int(
                        "interaction_smoothing_rounds", 10, 75
                    ),
                    # Tree structure parameters
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
                    "max_leaves": trial.suggest_int("max_leaves", 3, 8),
                    "min_hessian": trial.suggest_float("min_hessian", 1e-4, 1e-2, log=True),
                    # L1/L2 regularization
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
                    # Early stopping parameters - MUCH more conservative
                    "max_rounds": trial.suggest_int("max_rounds", 500, 3000),
                    "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 25, 100),
                    "early_stopping_tolerance": trial.suggest_float(
                        "early_stopping_tolerance", 1e-5, 1e-3, log=True
                    ),
                    # Categorical handling parameters
                    "gain_scale": trial.suggest_float("gain_scale", 1.0, 5.0),
                    "min_cat_samples": trial.suggest_int("min_cat_samples", 10, 25),
                    "cat_smooth": trial.suggest_float("cat_smooth", 5.0, 20.0),
                    # Missing value handling
                    "missing": trial.suggest_categorical("missing", ["separate", "low", "high"]),
                    # Fixed parameters
                    "n_jobs": self.threads,
                    "random_state": self.random_state,
                }
        elif self.model_type == "lightgbm":
            boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])

            # Base parameters
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": boosting_type,
                "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "num_leaves": trial.suggest_int("num_leaves", 10, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
                "random_state": self.random_state,
                "verbosity": -1,
                "force_col_wise": True,
                "n_jobs": self.threads,
            }

            # Add boosting-specific parameters
            if boosting_type == "goss":
                # GOSS-specific parameters (no bagging)
                params.update(
                    {
                        "top_rate": trial.suggest_float("top_rate", 0.1, 0.5),
                        "other_rate": trial.suggest_float("other_rate", 0.05, 0.2),
                    }
                )
            elif boosting_type == "dart":
                # DART-specific parameters
                params.update(
                    {
                        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                        "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.5),
                        "max_drop": trial.suggest_int("max_drop", 10, 100),
                        "skip_drop": trial.suggest_float("skip_drop", 0.4, 0.9),
                    }
                )
            else:  # gbdt
                # GBDT can use bagging
                params.update(
                    {
                        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                    }
                )

            return params
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def create_model(self, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        if self.model_type == "randomforest":
            return RandomForestClassifier(**params)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(**params)
        elif self.model_type == "ebm":
            return ExplainableBoostingClassifier(**params)
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def objective_function(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        train_dataset: PandasDataset = None,
        test_dataset: PandasDataset = None,
        dataset_version: str = None,
        user_tags: Dict[str, str] = None,
    ) -> float:
        """Optuna objective function for hyperparameter optimization."""
        params = self.get_hyperparameter_space(trial)
        with mlflow.start_run(nested=True, run_name=f"{self.model_type}_trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.set_tag("optuna_trial_number", str(trial.number))
            mlflow.set_tag("model_type", self.model_type)

            # Log all user tags to this trial
            if user_tags:
                mlflow.set_tags(user_tags)
                logger.debug(f"Set user tags for trial {trial.number}: {user_tags}")

            # Log dataset inputs for this trial
            if train_dataset:
                mlflow.log_input(train_dataset, context="training")
                logger.debug(f"Logged training dataset for trial {trial.number}")
            if test_dataset:
                mlflow.log_input(test_dataset, context="testing")
                logger.debug(f"Logged test dataset for trial {trial.number}")

            # Log dataset version as tag
            if dataset_version:
                mlflow.set_tag("dataset_version", dataset_version)
            else:
                mlflow.set_tag("dataset_version", "unknown")
                
            # Create model and time the training
            model = self.create_model(params)
            
            import time
            import signal
            
            # Set up timeout for model training (especially important for EBM with large datasets)
            timeout_seconds = self.trial_timeout
            if self.model_type == "ebm" and hasattr(self, '_feature_count') and self._feature_count and self._feature_count > 5000:
                # For large datasets, use 1.5x the configured timeout
                timeout_seconds = int(self.trial_timeout * 1.5)
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Model training exceeded {timeout_seconds} seconds")
            
            # Use alarm signal for timeout (Unix/Linux only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                
                fit_start = time.time()
                model.fit(X_train, y_train)
                fit_duration = time.time() - fit_start
                
                signal.alarm(0)  # Cancel the alarm
                
                # Log training time for visibility
                mlflow.log_metric("fit_seconds", fit_duration)
                logger.info(f"Trial {trial.number} model fit completed in {fit_duration:.1f}s")
                
            except TimeoutError as e:
                logger.warning(f"Trial {trial.number} timed out after {timeout_seconds}s: {e}")
                # Report a very poor score to Optuna so it knows this trial failed
                raise optuna.TrialPruned()
            except Exception as e:
                logger.error(f"Trial {trial.number} failed during training: {e}")
                raise optuna.TrialPruned()
            finally:
                signal.alarm(0)  # Ensure alarm is always cancelled
            
            # Time prediction as well
            predict_start = time.time()
            y_pred = model.predict(X_test)
            predict_duration = time.time() - predict_start
            mlflow.log_metric("predict_seconds", predict_duration)
            
            # EBM does not have predict_proba for multiclass, but for binary it does
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(y_test, y_pred, y_pred_proba)
            test_metrics = {f"test_{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(test_metrics)
            # Optionally reduce heavy artifact logging during trials
            if not self.light_artifacts:
                MetricsCalculator.log_curves_and_confusion(
                    y_test, y_pred, y_pred_proba, "test", trial.number
                )
                feature_names = (
                    X_train.columns.tolist()
                    if hasattr(X_train, "columns")
                    else [f"feature_{i}" for i in range(X_train.shape[1])]
                )
                FeatureImportanceAnalyzer.analyze_and_log(
                    model, feature_names, self.model_type, "trial", trial.number
                )

            # Calculate optimization score
            # Note: This uses simple averaging of metrics. For true multi-objective optimization,
            # Optuna supports returning a tuple of objectives with different directions.
            # Since all our metrics aim to be maximized, averaging is suitable here.
            optimization_score = np.mean([metrics[metric] for metric in self.optimization_metrics])
            return optimization_score


class OptunaStudyManager:
    """Manages Optuna study creation and optimization."""

    def __init__(self, study_name: str, storage_url: str, direction: str):
        self.study_name = study_name
        self.storage_url = storage_url
        self.direction = direction

    def create_study(self) -> optuna.Study:
        """Create or load existing Optuna study with pruner for efficiency."""
        logger.info(f"Creating/Loading study: {self.study_name}")

        # Add a pruner for efficiency - MedianPruner stops unpromising trials early
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Number of trials before pruning starts
            n_warmup_steps=10,  # Number of steps before pruning is allowed
            interval_steps=1,  # Pruning interval
        )

        # Use TPESampler for better hyperparameter search
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,  # Random sampling for first 10 trials
            n_ei_candidates=24,  # Number of candidates for expected improvement
            seed=42,  # For reproducibility
        )

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage_url,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )

        logger.info(
            f"Study configured with {type(pruner).__name__} pruner and {type(sampler).__name__} sampler"
        )

        return study

    def log_study_visualizations(self, study: optuna.Study) -> None:
        """Log Optuna study visualizations to MLflow."""
        if not optuna.visualization.is_available():
            logger.warning(
                "Optuna visualization not available. Install with: pip install optuna[visualization]"
            )
            return

        try:
            # Optimization history
            fig_history = optuna.visualization.plot_optimization_history(study)
            mlflow.log_figure(fig_history, "optuna_plots/optimization_history.html")

            # Parameter importance
            if len(study.trials) > 1:
                fig_importance = optuna.visualization.plot_param_importances(study)
                mlflow.log_figure(fig_importance, "optuna_plots/param_importances.html")

            # Slice plot
            if len(study.trials) > 1:
                fig_slice = optuna.visualization.plot_slice(study)
                mlflow.log_figure(fig_slice, "optuna_plots/slice_plot.html")

            logger.info("Optuna visualizations logged successfully")

        except Exception as e:
            logger.warning(f"Could not generate some Optuna visualizations: {e}")


class FeatureImportanceAnalyzer:
    """Handles feature importance analysis and visualization, including EBM."""

    @staticmethod
    def analyze_and_log(
        model,
        feature_names: List[str],
        model_type: str,
        prefix: str = "final",
        trial_number: Optional[int] = None,
    ) -> None:
        """Analyze feature importance and log results with percentages. Supports EBM with built-in explanations."""
        import io

        import numpy as np
        import pandas as pd

        # Extract feature importances based on model type
        if model_type.lower() == "ebm":
            # Use EBM's built-in explanation methods
            try:
                # Try to get global explanation from EBM
                if hasattr(model, "explain_global"):
                    global_explanation = model.explain_global()
                    if hasattr(global_explanation, "data"):
                        # Extract feature importances from global explanation
                        feature_data = global_explanation.data()
                        if "names" in feature_data and "scores" in feature_data:
                            ebm_feature_names = feature_data["names"]
                            ebm_importances = np.array(feature_data["scores"])

                            # Map EBM feature names to our feature names
                            importance_dict = {name: 0.0 for name in feature_names}
                            for ebm_name, ebm_imp in zip(ebm_feature_names, ebm_importances):
                                # Handle interaction terms (skip them for individual feature importance)
                                if " x " not in str(ebm_name) and ebm_name in feature_names:
                                    importance_dict[ebm_name] = abs(ebm_imp)  # Use absolute value

                            importances = np.array(
                                [importance_dict[name] for name in feature_names]
                            )
                            logger.info(
                                "Successfully extracted EBM feature importance using explain_global()"
                            )
                        else:
                            raise AttributeError("Global explanation data missing required keys")
                    else:
                        raise AttributeError("Global explanation missing data method")
                else:
                    raise AttributeError("EBM model missing explain_global method")

            except Exception as e:
                logger.warning(f"Could not use EBM explain_global method: {e}")
                # Fallback to traditional EBM attribute extraction
                try:
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        logger.info("Using EBM feature_importances_ attribute as fallback")
                    elif hasattr(model, "feature_groups_") and hasattr(model, "term_importances_"):
                        # EBM: sum term importances for each feature
                        term_importances = np.array(model.term_importances_)
                        feature_importance_dict = {name: 0.0 for name in feature_names}
                        for group, imp in zip(model.feature_groups_, term_importances):
                            for idx in group:
                                if idx < len(feature_names):  # Safety check
                                    feature_importance_dict[feature_names[idx]] += abs(imp)
                        importances = np.array(
                            [feature_importance_dict[name] for name in feature_names]
                        )
                        logger.info("Using EBM term_importances_ as fallback")
                    else:
                        raise AttributeError("EBM model missing required attributes")
                except Exception as fallback_e:
                    logger.error(
                        f"All EBM feature importance extraction methods failed: {fallback_e}"
                    )
                    return

        elif hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, XGBoost, LightGBM, etc.)
            importances = model.feature_importances_
        else:
            logger.warning(f"Model type {model_type} does not support feature importance.")
            return
        # Calculate percentages
        if importances.sum() > 0:
            importance_percentages = (importances / importances.sum()) * 100
        else:
            importance_percentages = np.zeros_like(importances)
        # Create DataFrame
        feat_imp_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "percentage": importance_percentages,
            }
        ).sort_values("importance", ascending=False)
        # Create enhanced visualization
        FeatureImportanceAnalyzer._create_importance_plot(
            feat_imp_df, model_type, prefix, trial_number
        )
        # Log CSV with percentages directly to MLflow
        suffix = f"_trial_{trial_number}" if trial_number is not None else ""
        csv_filename = f"feature_importances_{prefix}{suffix}.csv"
        csv_buffer = io.StringIO()
        feat_imp_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        mlflow.log_text(csv_content, f"plots/{csv_filename}")
        # Log top feature importance as metrics
        FeatureImportanceAnalyzer._log_top_features_as_metrics(feat_imp_df, prefix, trial_number)

        # For EBM, log additional explainability artifacts
        if model_type.lower() == "ebm":
            FeatureImportanceAnalyzer._log_ebm_explanations(model, prefix, trial_number)

        # Store top feature importances for stability analysis (for trials only)
        if trial_number is not None and prefix == "trial":
            try:
                top_5_features = feat_imp_df.head(5)
                # Format as "feature1:importance1,feature2:importance2,..."
                top_features_str = ",".join(
                    [
                        f"{row['feature']}:{row['importance']:.6f}"
                        for _, row in top_5_features.iterrows()
                    ]
                )

                # Store in current MLflow run's tags (accessible via Optuna trial)
                mlflow.set_tag(f"top_features_trial_{trial_number}", top_features_str)
                logger.debug(f"Stored top 5 features for trial {trial_number} stability analysis")
            except Exception as e:
                logger.debug(f"Could not store features for stability analysis: {e}")

        logger.info(f"Feature importance analysis completed for {model_type} ({prefix})")

    @staticmethod
    def _create_importance_plot(
        feat_imp_df: pd.DataFrame,
        model_type: str,
        prefix: str,
        trial_number: Optional[int] = None,
    ) -> None:
        """Create enhanced feature importance plot with percentages."""
        top_n = min(50, len(feat_imp_df))
        top_features = feat_imp_df.head(top_n)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(10, top_n * 0.4)))

        # Plot 1: Absolute importance values
        bars1 = ax1.barh(
            range(top_n), top_features["importance"][::-1], color="steelblue", alpha=0.8
        )
        ax1.set_yticks(range(top_n))
        ax1.set_yticklabels(top_features["feature"][::-1], fontsize=10)
        ax1.set_xlabel("Importance Value", fontsize=12)
        ax1.set_title(f"Top {top_n} Feature Importances - Absolute Values", fontsize=14)
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, top_features["importance"][::-1])):
            ax1.text(
                bar.get_width() + max(top_features["importance"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        # Plot 2: Percentage importance
        bars2 = ax2.barh(
            range(top_n), top_features["percentage"][::-1], color="darkgreen", alpha=0.8
        )
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_features["feature"][::-1], fontsize=10)
        ax2.set_xlabel("Importance Percentage (%)", fontsize=12)
        ax2.set_title(f"Top {top_n} Feature Importances - Percentages", fontsize=14)
        ax2.grid(axis="x", alpha=0.3)

        # Add percentage labels on bars
        for i, (bar, val) in enumerate(zip(bars2, top_features["percentage"][::-1])):
            ax2.text(
                bar.get_width() + max(top_features["percentage"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center",
                fontsize=9,
            )

        # Overall title
        suffix = f" - Trial {trial_number}" if trial_number is not None else ""
        fig.suptitle(f"Feature Importance Analysis ({model_type}){suffix}", fontsize=16, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save and log directly to MLflow without writing to disk
        trial_suffix = f"_trial_{trial_number}" if trial_number is not None else ""
        filename = f"feature_importances_{prefix}{trial_suffix}.png"
        mlflow.log_figure(fig, f"plots/{filename}")
        plt.close(fig)

    @staticmethod
    def _log_top_features_as_metrics(
        feat_imp_df: pd.DataFrame, prefix: str, trial_number: Optional[int] = None
    ) -> None:
        """Log top feature importances as MLflow metrics."""
        top_5 = feat_imp_df.head(5)

        metrics_dict = {}
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            feature_name = row["feature"][:20]  # Truncate long names
            metrics_dict[f"{prefix}_top_{i}_feature_importance"] = row["importance"]
            metrics_dict[f"{prefix}_top_{i}_feature_percentage"] = row["percentage"]
            # Log feature name as tag (since metrics must be numeric)
            mlflow.set_tag(f"{prefix}_top_{i}_feature_name", feature_name)

        # Log cumulative importance of top features
        metrics_dict[f"{prefix}_top_5_cumulative_percentage"] = top_5["percentage"].sum()
        metrics_dict[f"{prefix}_top_10_cumulative_percentage"] = feat_imp_df.head(10)[
            "percentage"
        ].sum()

        mlflow.log_metrics(metrics_dict)

    @staticmethod
    def _log_ebm_explanations(model, prefix: str, trial_number: Optional[int] = None) -> None:
        """Log EBM-specific explanations and visualizations."""
        try:
            if not hasattr(model, "explain_global"):
                logger.warning("EBM model does not have explain_global method")
                return

            global_explanation = model.explain_global()
            suffix = f"_trial_{trial_number}" if trial_number is not None else ""

            # Log global explanation data as JSON
            if hasattr(global_explanation, "data"):
                explanation_data = global_explanation.data()
                import json

                json_content = json.dumps(explanation_data, indent=2, default=str)
                mlflow.log_text(json_content, f"plots/ebm_global_explanation_{prefix}{suffix}.json")

            # Try to create EBM's built-in visualization
            try:
                # Check if plotly is available for EBM visualizations
                try:
                    import plotly

                    plotly_available = True
                except ImportError:
                    plotly_available = False
                    logger.debug("Plotly not available for EBM visualizations")

                if plotly_available and hasattr(global_explanation, "visualize"):
                    # Create HTML visualization
                    html_viz = global_explanation.visualize()
                    if html_viz:
                        mlflow.log_text(
                            str(html_viz), f"plots/ebm_global_viz_{prefix}{suffix}.html"
                        )
                        logger.info(f"EBM global visualization logged for {prefix}")
                elif not plotly_available:
                    logger.info(
                        "EBM visualizations require plotly. Install with: pip install plotly"
                    )

            except Exception as viz_e:
                logger.warning(f"Could not create EBM visualization: {viz_e}")

            # Log feature types and interactions info
            try:
                if hasattr(model, "feature_types_") and hasattr(model, "feature_names_"):
                    feature_info = {
                        "feature_names": (
                            list(model.feature_names_)
                            if hasattr(model.feature_names_, "__iter__")
                            else [str(model.feature_names_)]
                        ),
                        "feature_types": (
                            list(model.feature_types_)
                            if hasattr(model.feature_types_, "__iter__")
                            else [str(model.feature_types_)]
                        ),
                    }
                    if hasattr(model, "interactions"):
                        feature_info["interactions"] = model.interactions

                    import json

                    feature_info_json = json.dumps(feature_info, indent=2, default=str)
                    mlflow.log_text(
                        feature_info_json,
                        f"plots/ebm_feature_info_{prefix}{suffix}.json",
                    )
                    logger.info(f"EBM feature info logged for {prefix}")

            except Exception as info_e:
                logger.warning(f"Could not log EBM feature info: {info_e}")

        except Exception as e:
            logger.warning(f"Failed to log EBM explanations: {e}")

    @staticmethod
    def create_feature_stability_plot(study: optuna.Study, model_type: str) -> None:
        """Create a plot showing feature importance stability across trials."""
        try:
            # Get all completed trials
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]

            if len(completed_trials) < 3:
                logger.info("Not enough completed trials for stability analysis")
                return

            # Extract feature importances from MLflow tags
            # Note: This requires querying MLflow runs, which is complex in this context.
            # For now, we'll focus on optimization convergence analysis.
            trial_importances = {}
            all_features = set()

            # Try to get feature data (this would require MLflow client integration)
            # For now, we'll skip detailed feature stability and show convergence
            logger.info("Feature importance stability analysis requires MLflow client integration")
            logger.info("Showing optimization convergence instead")

            if not trial_importances:
                # Fallback: create optimization score stability plot
                logger.info(
                    "No feature importance data found in trials, showing optimization score stability"
                )
                fig, ax = plt.subplots(figsize=(12, 8))

                trial_numbers = [t.number for t in completed_trials]
                trial_values = [t.value for t in completed_trials]

                ax.plot(trial_numbers, trial_values, "b-", alpha=0.7, linewidth=2)
                ax.scatter(trial_numbers, trial_values, c="darkblue", s=50, alpha=0.8)
                ax.set_xlabel("Trial Number", fontsize=12)
                ax.set_ylabel("Optimization Score", fontsize=12)
                ax.set_title(f"Optimization Score Convergence ({model_type})", fontsize=14)
                ax.grid(True, alpha=0.3)

                # Add trend line and running average
                if len(trial_numbers) > 1:
                    z = np.polyfit(trial_numbers, trial_values, 1)
                    p = np.poly1d(z)
                    ax.plot(
                        trial_numbers,
                        p(trial_numbers),
                        "r--",
                        alpha=0.8,
                        label=f"Trend (slope: {z[0]:.6f})",
                    )

                    # Running average
                    window_size = min(10, len(trial_values) // 3)
                    if window_size > 1:
                        running_avg = (
                            pd.Series(trial_values).rolling(window=window_size, center=True).mean()
                        )
                        ax.plot(
                            trial_numbers,
                            running_avg,
                            "g-",
                            alpha=0.8,
                            linewidth=2,
                            label=f"Running Average (window={window_size})",
                        )

                    ax.legend()

                plt.tight_layout()
                mlflow.log_figure(fig, f"plots/optimization_convergence_{model_type.lower()}.png")
                plt.close(fig)
                return

            # Create actual feature importance stability plot
            top_features = list(all_features)[:10]  # Analyze top 10 features

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

            # Plot 1: Feature importance over trials
            trial_nums = sorted(trial_importances.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_features)))

            for i, feature in enumerate(top_features):
                importances = []
                for trial_num in trial_nums:
                    importances.append(trial_importances[trial_num].get(feature, 0))

                ax1.plot(
                    trial_nums,
                    importances,
                    "o-",
                    color=colors[i],
                    alpha=0.7,
                    label=feature[:20],
                    markersize=4,
                )

            ax1.set_xlabel("Trial Number")
            ax1.set_ylabel("Feature Importance")
            ax1.set_title(f"Feature Importance Stability Across Trials ({model_type})")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Feature importance variance (stability metric)
            feature_vars = {}
            feature_means = {}

            for feature in top_features:
                importances = [
                    trial_importances[trial_num].get(feature, 0) for trial_num in trial_nums
                ]
                feature_vars[feature] = np.var(importances)
                feature_means[feature] = np.mean(importances)

            # Sort by stability (lower variance = more stable)
            sorted_features = sorted(feature_vars.items(), key=lambda x: x[1])
            features_sorted = [f[0] for f in sorted_features]
            variances_sorted = [f[1] for f in sorted_features]

            bars = ax2.barh(
                range(len(features_sorted)),
                variances_sorted,
                color="lightcoral",
                alpha=0.7,
            )
            ax2.set_yticks(range(len(features_sorted)))
            ax2.set_yticklabels([f[:20] for f in features_sorted])
            ax2.set_xlabel("Importance Variance (lower = more stable)")
            ax2.set_title("Feature Importance Stability Ranking")
            ax2.grid(True, alpha=0.3, axis="x")

            # Add variance values on bars
            for i, (bar, var) in enumerate(zip(bars, variances_sorted)):
                ax2.text(
                    bar.get_width() + max(variances_sorted) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{var:.4f}",
                    va="center",
                    fontsize=9,
                )

            plt.tight_layout()
            mlflow.log_figure(fig, f"plots/feature_stability_{model_type.lower()}.png")
            plt.close(fig)

            # Log stability metrics
            stability_metrics = {}
            for feature, variance in feature_vars.items():
                clean_name = feature.replace(" ", "_").replace("-", "_")[:20]
                stability_metrics[f"feature_stability_var_{clean_name}"] = variance
                stability_metrics[f"feature_stability_mean_{clean_name}"] = feature_means[feature]

            mlflow.log_metrics(stability_metrics)
            logger.info(f"Feature importance stability analysis completed for {model_type}")

        except Exception as e:
            logger.warning(f"Could not create feature stability plot: {e}")


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Setup MLflow tracking and experiment."""
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Using existing MLflow experiment '{experiment_name}' with ID: {experiment_id}"
        )

    mlflow.set_experiment(experiment_name)
    return experiment_id


def parse_tags(tags_list: List[str]) -> Dict[str, str]:
    """Parse key=value tag pairs into a dictionary."""
    tags = {}
    if tags_list:
        for tag in tags_list:
            if "=" in tag:
                key, value = tag.split("=", 1)  # Split on first '=' only
                tags[key.strip()] = value.strip()
            else:
                logger.warning(f"Invalid tag format: '{tag}'. Expected 'key=value' format.")
    return tags


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--train-data",
        required=True,
        type=str,
        help="Path to training dataset (parquet format)",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        type=str,
        help="Path to test dataset (parquet format)",
    )
    parser.add_argument(
        "--holdout-data",
        required=True,
        type=str,
        help="Path to holdout dataset (parquet format)",
    )
    parser.add_argument("--target-column", default="y", type=str, help="Name of the target column")

    # Model arguments
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["randomforest", "xgboost", "ebm", "lightgbm"],
        help="Type of model to optimize",
    )
    parser.add_argument(
        "--optimization-metrics",
        nargs="+",
        default=["precision"],
        choices=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        help="Metrics to optimize (will be averaged)",
    )
    parser.add_argument(
        "--optimization-direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="Direction of optimization",
    )

    # Optuna arguments
    parser.add_argument("--study-name", required=True, type=str, help="Name for the Optuna study")
    parser.add_argument(
        "--storage-url",
        default="sqlite:///optuna.db",
        help="Database URL for Optuna storage",
    )
    parser.add_argument("--n-trials", default=100, type=int, help="Number of Optuna trials to run")
    parser.add_argument("--timeout", type=int, help="Timeout for optimization in seconds")
    parser.add_argument("--trial-timeout", type=int, default=7200, help="Timeout per individual trial in seconds (default: 7200 = 2 hours)")

    # MLflow arguments
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", required=True, type=str, help="MLflow experiment name")

    # Other arguments
    parser.add_argument(
        "--random-state", default=42, type=int, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--project-name",
        default="ML Hyperparameter Optimization",
        help="Project name for MLflow tags",
    )
    parser.add_argument(
        "--dataset-version",
        default="v1.0",
        help="Dataset version number for MLflow dataset version parameter",
    )
    parser.add_argument(
        "--dataset-suffix",
        default="training_dataset",
        help="Dataset suffix for MLflow tags and base name",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset configuration name (e.g., redux_filter_3) for model registration",
    )

    # Feature Selection Arguments
    parser.add_argument(
        "--analyze-correlations",
        action="store_true",
        help="If set, analyze and drop highly correlated features before optimization.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for dropping features.",
    )
    parser.add_argument(
        "--drop-features-file",
        type=str,
        help="Path to a text file with a list of features to drop (one per line).",
    )

    # New tags argument
    parser.add_argument(
        "--tags",
        nargs="+",
        type=str,
        help="Additional tags in key=value format (e.g., --tags experiment_type=test batch_id=001)",
    )

    # Thread configuration
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for model training and optimization (default: 1)",
    )

    # Debug arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed troubleshooting",
    )

    # Top models training
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top models to train and evaluate on holdout data (default: 1)",
    )

    # Light artifacts mode
    parser.add_argument(
        "--light-artifacts",
        action="store_true",
        help="Log minimal artifacts during trials (metrics only)",
    )

    return parser


def train_and_evaluate_top_models(
    study: optuna.Study,
    optimizer: ModelOptimizer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    holdout_dataset,
    top_n: int,
    model_type: str,
    optimization_metrics: List[str],
    experiment_name: str = None,
    dataset_name: str = None,
    dataset_tags: Dict[str, str] = None,
) -> int:
    """
    Train and evaluate the top N models from the Optuna study on the holdout dataset.

    Returns:
        int: Number of models actually trained (may be less than top_n if fewer trials exist)
    """
    if not study.trials:
        logger.warning("No completed trials found in study")
        return 0

    # Get top N trials sorted by objective value
    # For multi-objective optimization, we'll use the first objective or aggregated value
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logger.warning("No completed trials found in study")
        return 0

    # Sort trials by objective value
    # Optuna stores values as a list for multi-objective optimization
    # For single objective, it's still a list with one element
    def get_sort_key(trial):
        if trial.values is None:
            return (
                float("-inf")
                if study.direction == optuna.study.StudyDirection.MAXIMIZE
                else float("inf")
            )
        # Use the first objective value, or average if multiple objectives
        if len(trial.values) == 1:
            return trial.values[0]
        else:
            # For multiple objectives, use the average
            return sum(trial.values) / len(trial.values)

    # Sort by objective value
    ascending = study.direction == optuna.study.StudyDirection.MINIMIZE
    sorted_trials = sorted(completed_trials, key=get_sort_key, reverse=not ascending)

    # Limit to top N
    actual_top_n = min(top_n, len(sorted_trials))
    top_trials = sorted_trials[:actual_top_n]

    logger.info(
        f"Training and evaluating top {actual_top_n} models from {len(completed_trials)} total completed trials"
    )

    models_trained = 0

    for i, trial in enumerate(top_trials, 1):
        try:
            # Get trial information
            trial_number = trial.number
            trial_value = get_sort_key(trial)

            logger.info(
                f"Training model {i}/{actual_top_n} (Trial #{trial_number}, Score: {trial_value:.4f})"
            )

            # Create nested MLflow run for this top model
            with mlflow.start_run(nested=True, run_name=f"top_{i}_model_trial_{trial_number}"):
                # Log trial information
                mlflow.log_param("top_model_rank", i)
                mlflow.log_param("original_trial_number", trial_number)
                mlflow.log_param("optimization_score", trial_value)
                mlflow.log_params(trial.params)

                # Log dataset-specific tags to the model run
                if dataset_tags:
                    mlflow.set_tags(dataset_tags)
                    logger.debug(f"Applied dataset tags to top {i} model: {dataset_tags}")

                # Log individual objective values if multiple
                if trial.values and len(trial.values) > 1:
                    for j, value in enumerate(trial.values):
                        mlflow.log_param(f"objective_{j}_value", value)

                # Create and train the model
                model = optimizer.create_model(trial.params)
                logger.debug(f"Training model with parameters: {trial.params}")
                model.fit(X_train, y_train)

                # Evaluate on holdout set
                y_holdout_pred = model.predict(X_holdout)
                if hasattr(model, "predict_proba"):
                    y_holdout_pred_proba = model.predict_proba(X_holdout)[:, 1]
                else:
                    y_holdout_pred_proba = y_holdout_pred

                # Calculate and log metrics
                holdout_metrics = MetricsCalculator.calculate_metrics(
                    y_holdout, y_holdout_pred, y_holdout_pred_proba
                )

                # Log metrics with holdout prefix
                mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})

                # Log the optimization metrics specifically
                for j, metric in enumerate(optimization_metrics):
                    if metric in holdout_metrics:
                        mlflow.log_metric(f"optimization_metric_{metric}", holdout_metrics[metric])

                # Log curves and confusion matrix
                MetricsCalculator.log_curves_and_confusion(
                    y_holdout,
                    y_holdout_pred,
                    y_holdout_pred_proba,
                    f"holdout_top_{i}",
                    trial_number,
                )

                # Log feature importance for this model
                feature_names = X_train.columns.tolist()
                FeatureImportanceAnalyzer.analyze_and_log(
                    model, feature_names, model_type, f"top_{i}", trial_number
                )

                # Log model as MLflow model (not just artifact)
                try:
                    model_name = f"top_{i}_model"

                    # Use dataset name for model registration, fallback to experiment name, then model type
                    base_model_name = dataset_name or experiment_name or model_type
                    registered_model_name = f"{base_model_name}_top_{i}"

                    # Prepare model description with key information
                    model_description = f"Top {i} {model_type} model for {base_model_name}"
                    if dataset_tags:
                        feature_info = []
                        # Note: CPAT features have been removed from the pipeline
                        if "features_ss" in dataset_tags:
                            feature_info.append(f"SS: {dataset_tags['features_ss']}")
                        if "features_filter" in dataset_tags:
                            feature_info.append(f"Filter: {dataset_tags['features_filter']}")
                        if "dataset_imbalance_technique" in dataset_tags:
                            feature_info.append(
                                f"Imbalance: {dataset_tags['dataset_imbalance_technique']}"
                            )
                        if feature_info:
                            model_description += f" | {' | '.join(feature_info)}"

                    # Add optimization metrics to description
                    opt_metrics_str = ", ".join(
                        [
                            f"{metric}: {holdout_metrics.get(metric, 'N/A'):.4f}"
                            for metric in optimization_metrics
                        ]
                    )
                    model_description += f" | Metrics: {opt_metrics_str}"

                    if model_type == "randomforest":
                        # Create input example for model signature
                        input_example = X_train.head(1)

                        model_info = mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path=model_name,
                            input_example=input_example,
                            registered_model_name=registered_model_name,
                            signature=mlflow.models.infer_signature(X_train, y_train),
                        )
                        logger.debug(
                            f"Logged RandomForest model for top {i} model to MLflow Model Registry as '{registered_model_name}'"
                        )

                    elif model_type == "xgboost":
                        # Create input example for model signature
                        input_example = X_train.head(1)

                        model_info = mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path=model_name,
                            input_example=input_example,
                            registered_model_name=registered_model_name,
                            signature=mlflow.models.infer_signature(X_train, y_train),
                        )
                        logger.debug(
                            f"Logged XGBoost model for top {i} model to MLflow Model Registry as '{registered_model_name}'"
                        )

                    elif model_type == "lightgbm":
                        # Create input example for model signature
                        input_example = X_train.head(1)

                        model_info = mlflow.lightgbm.log_model(
                            lgb_model=model,
                            artifact_path=model_name,
                            input_example=input_example,
                            registered_model_name=registered_model_name,
                            signature=mlflow.models.infer_signature(X_train, y_train),
                        )
                        logger.debug(
                            f"Logged LightGBM model for top {i} model to MLflow Model Registry as '{registered_model_name}'"
                        )

                    elif model_type == "ebm":
                        # EBM models can be logged as Python function models
                        # since MLflow doesn't have native EBM support
                        input_example = X_train.head(1)

                        # Create a custom Python model wrapper for EBM
                        class EBMModelWrapper(mlflow.pyfunc.PythonModel):
                            def __init__(self, ebm_model):
                                self.ebm_model = ebm_model

                            def predict(self, context, model_input):
                                return self.ebm_model.predict_proba(model_input)[:, 1]

                        # Log as pyfunc model
                        model_info = mlflow.pyfunc.log_model(
                            artifact_path=model_name,
                            python_model=EBMModelWrapper(model),
                            input_example=input_example,
                            registered_model_name=registered_model_name,
                            signature=mlflow.models.infer_signature(X_train, y_train),
                            conda_env={
                                "channels": ["defaults", "conda-forge"],
                                "dependencies": [
                                    "python=3.8",
                                    "pip",
                                    {
                                        "pip": [
                                            "interpret",
                                            "pandas",
                                            "numpy",
                                            "scikit-learn",
                                        ]
                                    },
                                ],
                                "name": "ebm_env",
                            },
                        )
                        logger.debug(
                            f"Logged EBM model for top {i} model to MLflow Model Registry as pyfunc '{registered_model_name}'"
                        )

                    # Update the registered model version with metadata and tags
                    try:
                        client = mlflow.tracking.MlflowClient()

                        # Get the latest version number for this model
                        latest_versions = client.get_latest_versions(registered_model_name)
                        if latest_versions:
                            latest_version = latest_versions[0].version

                            # Update model version description
                            client.update_model_version(
                                name=registered_model_name,
                                version=latest_version,
                                description=model_description,
                            )

                            # Set tags on the model version
                            model_version_tags = {}

                            # Add dataset configuration tags
                            if dataset_tags:
                                for key, value in dataset_tags.items():
                                    model_version_tags[f"dataset_{key}"] = str(value)

                            # Add model performance tags
                            for metric_name, metric_value in holdout_metrics.items():
                                model_version_tags[f"holdout_{metric_name}"] = f"{metric_value:.4f}"

                            # Add optimization information tags
                            model_version_tags.update(
                                {
                                    "model_type": model_type,
                                    "optimization_metrics": ",".join(optimization_metrics),
                                    "optimization_score": f"{trial_value:.4f}",
                                    "trial_number": str(trial_number),
                                    "top_model_rank": str(i),
                                    "dataset_name": dataset_name or "unknown",
                                    "training_samples": str(len(X_train)),
                                    "feature_count": str(X_train.shape[1]),
                                }
                            )

                            # Set all tags on the model version
                            for tag_key, tag_value in model_version_tags.items():
                                try:
                                    client.set_model_version_tag(
                                        name=registered_model_name,
                                        version=latest_version,
                                        key=tag_key,
                                        value=tag_value,
                                    )
                                except Exception as tag_e:
                                    logger.warning(
                                        f"Failed to set tag {tag_key}={tag_value} on model version: {tag_e}"
                                    )

                            logger.info(
                                f"✅ Updated model version {latest_version} with {len(model_version_tags)} tags and description"
                            )

                            # Also set tags on the registered model itself (model-level tags)
                            try:
                                model_level_tags = {
                                    "model_type": model_type,
                                    "dataset_name": dataset_name or "unknown",
                                    "optimization_metrics": ",".join(optimization_metrics),
                                }

                                for tag_key, tag_value in model_level_tags.items():
                                    client.set_registered_model_tag(
                                        name=registered_model_name,
                                        key=tag_key,
                                        value=tag_value,
                                    )

                                logger.debug(f"Set {len(model_level_tags)} model-level tags")

                            except Exception as model_tag_e:
                                logger.warning(f"Failed to set model-level tags: {model_tag_e}")

                        else:
                            logger.warning(
                                f"Could not find latest version for model {registered_model_name}"
                            )

                    except Exception as registry_e:
                        logger.warning(f"Failed to update model registry metadata: {registry_e}")
                        logger.debug("Registry update exception details:", exc_info=True)

                    # Log model metadata to the MLflow run as well
                    mlflow.log_param(f"model_{i}_registered_name", registered_model_name)
                    mlflow.log_param(f"model_{i}_artifact_path", model_name)
                    mlflow.log_param(f"model_{i}_dataset_name", dataset_name or "unknown")
                    mlflow.log_param(f"model_{i}_description", model_description)

                    logger.info(
                        f"✅ Successfully logged top {i} model to MLflow Model Registry as '{registered_model_name}'"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to log model to MLflow Model Registry for top {i} model: {e}"
                    )
                    logger.debug("Model logging exception details:", exc_info=True)

                    # Fallback to basic artifact logging if MLflow model logging fails
                    try:
                        if model_type == "randomforest" and joblib is not None:
                            # Use tempfile for safe temporary file handling
                            with tempfile.NamedTemporaryFile(
                                suffix=".joblib", delete=False
                            ) as tmp_file:
                                model_path = tmp_file.name
                            try:
                                joblib.dump(model, model_path)
                                mlflow.log_artifact(model_path)
                                logger.debug(
                                    f"Fallback: Saved RandomForest as artifact for top {i} model"
                                )
                            finally:
                                if os.path.exists(model_path):
                                    os.unlink(model_path)
                        elif model_type in ["xgboost", "lightgbm"]:
                            with tempfile.NamedTemporaryFile(
                                suffix=".json", delete=False
                            ) as tmp_file:
                                model_path = tmp_file.name
                            try:
                                model.save_model(model_path)
                                mlflow.log_artifact(model_path)
                                logger.debug(
                                    f"Fallback: Saved {model_type} as artifact for top {i} model"
                                )
                            finally:
                                if os.path.exists(model_path):
                                    os.unlink(model_path)
                    except Exception as fallback_e:
                        logger.warning(
                            f"Even fallback artifact logging failed for top {i} model: {fallback_e}"
                        )

                models_trained += 1
                logger.info(f"✅ Completed evaluation of top {i} model (Trial #{trial_number})")

        except Exception as e:
            logger.error(f"❌ Failed to train/evaluate top {i} model: {e}")
            logger.debug("Exception details:", exc_info=True)
            continue

    logger.info(f"Successfully trained and evaluated {models_trained}/{actual_top_n} top models")
    return models_trained


def optimize_hyperparameters(
    train_data_path: str,
    test_data_path: str,
    holdout_data_path: str,
    model_type: str,
    study_name: str,
    experiment_name: str,
    target_column: str = "y",
    optimization_metrics: Optional[List[str]] = None,
    optimization_direction: str = "maximize",
    n_trials: int = 100,
    mlflow_uri: str = "sqlite:///mlflow.db",
    storage_url: str = "sqlite:///optuna.db",
    random_state: int = 42,
    tags: Optional[List[str]] = None,
    debug: bool = False,
    project_name: str = "ML Hyperparameter Optimization",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for ML models.

    This function provides a Python API for hyperparameter optimization using Optuna
    and MLflow tracking. It optimizes model hyperparameters and evaluates performance
    on train, test, and holdout datasets.

    Args:
        train_data_path: Path to training dataset (parquet format)
        test_data_path: Path to test dataset (parquet format)
        holdout_data_path: Path to holdout dataset (parquet format)
        model_type: Type of model ('randomforest', 'xgboost', 'lightgbm', 'ebm')
        study_name: Name for the Optuna study
        experiment_name: MLflow experiment name
        target_column: Name of target column in datasets
        optimization_metrics: Metrics to optimize. Defaults to ['precision']
        optimization_direction: 'maximize' or 'minimize'
        n_trials: Number of optimization trials
        mlflow_uri: MLflow tracking URI
        storage_url: Optuna storage database URL
        random_state: Random seed for reproducibility
        tags: List of tags in 'key=value' format
        debug: Enable debug logging
        project_name: Project name for MLflow tags
        **kwargs: Additional parameters (timeout, threads, top_n, etc.)

    Returns:
        Dict containing optimization results:
            - 'study': Completed Optuna study object
            - 'best_params': Best hyperparameters found
            - 'best_value': Best objective value achieved
            - 'models_trained': Number of top models trained on holdout data

    Raises:
        FileNotFoundError: If data files don't exist
        ValueError: If parameters are invalid
        RuntimeError: If optimization fails

    Example:
        >>> from src.optimizer.hyperparameter_optimizer import optimize_hyperparameters
        >>>
        >>> results = optimize_hyperparameters(
        ...     train_data_path="train.parquet",
        ...     test_data_path="test.parquet",
        ...     holdout_data_path="holdout.parquet",
        ...     model_type="randomforest",
        ...     study_name="rf_optimization",
        ...     experiment_name="experiment_1",
        ...     optimization_metrics=["precision", "f1"],
        ...     n_trials=100,
        ...     tags=["experiment_type=test", "batch_id=001"]
        ... )
        >>> print(f"Best parameters: {results['best_params']}")
        >>> print(f"Best score: {results['best_value']}")
    """
    if optimization_metrics is None:
        optimization_metrics = ["precision"]

    # Configure logging level based on debug flag
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        _load_dotenv()
        # Validate file paths exist
        DataLoader.validate_file_paths(train_data_path, test_data_path, holdout_data_path)

        # Set up MLflow
        experiment_id = setup_mlflow(mlflow_uri, experiment_name)

        # Load datasets (now with target_column parameter)
        train_df, test_df, holdout_df = DataLoader.load_datasets(
            train_data_path, test_data_path, holdout_data_path
        )

        # Apply feature selection if requested
        analyze_correlations = kwargs.get("analyze_correlations", False)
        correlation_threshold = kwargs.get("correlation_threshold", 0.95)
        drop_features_file = kwargs.get("drop_features_file")

        if analyze_correlations or drop_features_file:
            train_df, test_df, holdout_df = DataLoader.apply_feature_selection(
                train_df,
                test_df,
                holdout_df,
                target_column,
                analyze_correlations,
                correlation_threshold,
                drop_features_file,
            )

        # Prepare data
        X_train, y_train = DataLoader.prepare_features_and_targets(train_df, target_column)
        X_test, y_test = DataLoader.prepare_features_and_targets(test_df, target_column)
        X_holdout, y_holdout = DataLoader.prepare_features_and_targets(holdout_df, target_column)

        # Create datasets for MLflow with proper version handling
        dataset_version = kwargs.get("dataset_version", "v1.0")
        dataset_suffix = kwargs.get("dataset_suffix", "training_dataset")

        try:
            # Try the newer MLflow API with version parameter
            train_dataset = mlflow.data.from_pandas(
                train_df,
                source=train_data_path,
                name=f"{dataset_suffix}_train",
                version=dataset_version,
                targets=target_column,
            )
            test_dataset = mlflow.data.from_pandas(
                test_df,
                source=test_data_path,
                name=f"{dataset_suffix}_test",
                version=dataset_version,
                targets=target_column,
            )
            holdout_dataset = mlflow.data.from_pandas(
                holdout_df,
                source=holdout_data_path,
                name=f"{dataset_suffix}_holdout",
                version=dataset_version,
                targets=target_column,
            )
        except TypeError as e:
            if "version" in str(e):
                # Fallback to older MLflow API without version parameter
                logger.warning("MLflow version parameter not supported, using fallback")
                train_dataset = mlflow.data.from_pandas(
                    train_df, source=train_data_path, targets=target_column
                )
                test_dataset = mlflow.data.from_pandas(
                    test_df, source=test_data_path, targets=target_column
                )
                holdout_dataset = mlflow.data.from_pandas(
                    holdout_df, source=holdout_data_path, targets=target_column
                )
            else:
                raise

        # Initialize optimizer
        optimizer = ModelOptimizer(
            model_type=model_type,
            optimization_metrics=optimization_metrics,
            optimization_direction=optimization_direction,
            random_state=random_state,
            threads=kwargs.get("threads", 1),
            light_artifacts=kwargs.get("light_artifacts", False),
            trial_timeout=kwargs.get("trial_timeout", 7200),
        )
        
        # Set feature count for adaptive parameter selection
        optimizer._feature_count = X_train.shape[1]
        logger.info(f"Dataset has {optimizer._feature_count} features - using {'large dataset' if optimizer._feature_count > 5000 else 'standard'} EBM parameters")

        # Create and run study
        study_manager = OptunaStudyManager(study_name, storage_url, optimization_direction)
        study = study_manager.create_study()

        # Parse tags
        user_tags = parse_tags(tags or [])

        # Add default tags
        user_tags.update(
            {
                "project_name": project_name,
                "dataset_version": dataset_version,
                "optimization_metrics": ",".join(optimization_metrics),
                "optimization_direction": optimization_direction,
                "model_type": model_type,
                "study_name": study_name,
                "random_state": str(random_state),
            }
        )

        # Optimize
        study.optimize(
            lambda trial: optimizer.objective_function(
                trial,
                X_train,
                y_train,
                X_test,
                y_test,
                train_dataset,
                test_dataset,
                dataset_version,
                user_tags,
            ),
            n_trials=n_trials,
            timeout=kwargs.get("timeout"),
        )

        # Train top models
        top_n = kwargs.get("top_n", 1)
        models_trained = train_and_evaluate_top_models(
            study,
            optimizer,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            holdout_dataset,
            top_n,
            model_type,
            optimization_metrics,
            experiment_name,
            kwargs.get("dataset_name"),
            user_tags,
        )

        return {
            "study": study,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "models_trained": models_trained,
        }

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        raise


def main() -> int:
    """
    Main execution function for CLI.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Ensure env vars (e.g., AWS creds for MinIO) are available
    _load_dotenv()

    # Configure logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Parse additional tags
    additional_tags = parse_tags(args.tags) if args.tags else {}

    logger.info("Starting hyperparameter optimization...")
    logger.info(f"Configuration: {vars(args)}")
    logger.debug(f"Python path: {sys.path}")
    logger.debug(f"MLflow version: {mlflow.__version__}")

    try:
        # Load datasets
        train_df, test_df, holdout_df = DataLoader.load_datasets(
            args.train_data, args.test_data, args.holdout_data
        )
        logger.debug(
            f"Loaded datasets - Train: {train_df.shape}, Test: {test_df.shape}, Holdout: {holdout_df.shape}"
        )

        # Prepare features and targets
        X_train, y_train = DataLoader.prepare_features_and_targets(train_df, args.target_column)
        X_test, y_test = DataLoader.prepare_features_and_targets(test_df, args.target_column)
        X_holdout, y_holdout = DataLoader.prepare_features_and_targets(
            holdout_df, args.target_column
        )

        # Setup MLflow
        experiment_id = setup_mlflow(args.mlflow_uri, args.experiment_name)

        # Main MLflow run
        with mlflow.start_run(
            run_name="hyperparameter_optimization_main", experiment_id=experiment_id
        ) as main_run:
            logger.debug(f"Started main MLflow run: {main_run.info.run_id}")

            # Log dataset version as tag for compatibility
            mlflow.set_tag("dataset_version", args.dataset_version)

            # --- Feature Selection Step (before optimization) ---
            features_to_drop = []
            if args.analyze_correlations:
                features_to_drop = FeatureAnalyzer.analyze_and_log_correlation(
                    X_train,
                    args.correlation_threshold,
                    None,  # No longer need output_dir
                )
            elif args.drop_features_file:
                logger.info(f"Loading features to drop from {args.drop_features_file}")
                try:
                    with open(args.drop_features_file, "r") as f:
                        features_to_drop = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(features_to_drop)} features to drop from file")
                    # Log the drop features file as an artifact
                    mlflow.log_artifact(args.drop_features_file, "feature_selection")
                    # Log relevant information
                    mlflow.log_param("drop_features_file", args.drop_features_file)
                    mlflow.log_param("num_features_dropped", len(features_to_drop))
                except Exception as e:
                    logger.error(f"Error loading features to drop: {e}")
                    features_to_drop = []

            # Apply feature selection
            if features_to_drop:
                logger.info(f"Dropping {len(features_to_drop)} features before optimization")
                features_to_drop_in_data = [f for f in features_to_drop if f in X_train.columns]
                features_not_found = [f for f in features_to_drop if f not in X_train.columns]

                if features_not_found:
                    logger.warning(f"Features not found in dataset: {features_not_found}")

                X_train = X_train.drop(columns=features_to_drop_in_data)
                X_test = X_test.drop(columns=features_to_drop_in_data)
                X_holdout = X_holdout.drop(columns=features_to_drop_in_data)

                logger.info(f"Remaining features after selection: {X_train.shape[1]}")
                mlflow.log_param("num_features_remaining", X_train.shape[1])

            # Rebuild DataFrames with selected features for MLflow dataset creation
            train_df_selected = X_train.copy()
            train_df_selected[args.target_column] = y_train

            test_df_selected = X_test.copy()
            test_df_selected[args.target_column] = y_test

            holdout_df_selected = X_holdout.copy()
            holdout_df_selected[args.target_column] = y_holdout

            # Create MLflow datasets with selected features for proper tracking
            logger.debug("Creating MLflow datasets with selected features...")
            train_dataset, test_dataset, holdout_dataset = DataLoader.create_mlflow_datasets(
                train_df_selected,
                test_df_selected,
                holdout_df_selected,
                args.train_data,
                args.test_data,
                args.holdout_data,
                args.target_column,
                args.dataset_version,
                args.dataset_suffix,
            )

            # Log datasets at the main run level with selected features
            try:
                mlflow.log_input(train_dataset, context="training")
                mlflow.log_input(test_dataset, context="testing")
                mlflow.log_input(holdout_dataset, context="validation")
                logger.debug("Successfully logged all datasets to main MLflow run")
            except Exception as e:
                logger.warning(f"Failed to log datasets to MLflow: {e}")
                logger.debug("Continuing without dataset logging...")

            # Create optimizer
            optimizer = ModelOptimizer(
                args.model_type,
                args.optimization_metrics,
                args.optimization_direction,
                args.random_state,
                args.threads,
                args.light_artifacts,
                args.trial_timeout,
            )
            study_manager = OptunaStudyManager(
                args.study_name, args.storage_url, args.optimization_direction
            )

            # Create study
            study = study_manager.create_study()

            # Comprehensive user tags for MLflow
            user_tags = {
                "project_name": args.project_name,
                "dataset_version": args.dataset_version,
                "optimization_metrics": ",".join(args.optimization_metrics),
                "optimization_direction": args.optimization_direction,
                "model_type": args.model_type,
                "study_name": args.study_name,
                "dataset_suffix": args.dataset_suffix,
                "random_state": str(args.random_state),
                "n_trials": str(args.n_trials),
            }

            # Add feature selection related tags
            if args.analyze_correlations:
                user_tags["analyze_correlations"] = "true"
                user_tags["correlation_threshold"] = str(args.correlation_threshold)

            if args.drop_features_file:
                user_tags["drop_features_file"] = args.drop_features_file
                user_tags["num_features_dropped_from_file"] = str(len(features_to_drop))

            user_tags["num_features_used"] = str(X_train.shape[1])

            # Add any additional user-specified tags
            user_tags.update(additional_tags)

            # Log all tags at once
            mlflow.set_tags(user_tags)
            logger.debug(f"Set MLflow tags: {user_tags}")

            # Run optimization
            def objective_func(trial):
                return optimizer.objective_function(
                    trial,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    train_dataset,
                    test_dataset,
                    args.dataset_version,
                    user_tags,
                )

            study.optimize(objective_func, n_trials=args.n_trials, timeout=args.timeout)

            # Log best trial results
            logger.info("Optimization finished. Logging best trial results...")
            if not study.trials:
                logger.warning("No trials completed. Cannot train final models.")
                return

            best_trial = study.best_trial
            mlflow.log_params(best_trial.params)
            mlflow.log_metrics(
                {
                    f"best_{metric}": getattr(
                        best_trial.values[i] if len(best_trial.values) > i else 0,
                        "__float__",
                        lambda: 0,
                    )()
                    for i, metric in enumerate(args.optimization_metrics)
                }
            )
            mlflow.set_tag("best_trial_number", str(best_trial.number))

            # Train and evaluate top-n models
            logger.info(f"Training top {args.top_n} model(s) and evaluating on holdout set...")

            # Extract dataset-specific tags from user_tags for model registration
            dataset_specific_tags = {}
            if user_tags:
                # Include tags that are specific to the dataset configuration
                dataset_tag_prefixes = ["dataset_", "features_", "imbalance_"]
                for key, value in user_tags.items():
                    if any(key.startswith(prefix) for prefix in dataset_tag_prefixes):
                        dataset_specific_tags[key] = value
                # Also include some general tags that are important for models
                important_general_tags = [
                    "project_name",
                    "model_type",
                    "optimization_metrics",
                    "dataset_version",
                ]
                for key in important_general_tags:
                    if key in user_tags:
                        dataset_specific_tags[key] = user_tags[key]

            top_n_models_trained = train_and_evaluate_top_models(
                study,
                optimizer,
                X_train,
                y_train,
                X_holdout,
                y_holdout,
                holdout_dataset,
                args.top_n,
                args.model_type,
                args.optimization_metrics,
                args.experiment_name,
                args.dataset_name,
                dataset_specific_tags,
            )

            # Log metadata about top models training
            mlflow.log_param("top_n_models_requested", args.top_n)
            mlflow.log_param("top_n_models_trained", top_n_models_trained)
            mlflow.log_param("final_training_samples", len(X_train))
            mlflow.log_param("final_feature_count", X_train.shape[1])

            # Log the holdout dataset for final evaluation
            try:
                mlflow.log_input(holdout_dataset, context="final_evaluation")
                logger.debug("Logged holdout dataset for final evaluation")
            except Exception as e:
                logger.warning(f"Failed to log holdout dataset: {e}")

            # Log study visualizations and stability analysis
            study_manager.log_study_visualizations(study)
            FeatureImportanceAnalyzer.create_feature_stability_plot(study, args.model_type)

            logger.info("Script finished successfully!")
            return 0

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
