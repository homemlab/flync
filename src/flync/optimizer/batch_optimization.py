#!/usr/bin/env python3
"""
Batch Hyperparameter Optimization Script

Submits multiple hyperparameter optimization jobs with different combinations of:
- Model types (RandomForest, XGBoost, LightGBM, EBM)
- Optimization metrics (precision, recall, f1, roc_auc, pr_auc)
- Various parameter combinations

Based on the hyperparameter_optimizer.py script.

Features:
- YAML configuration file support for complex batch setups
- File path validation before job submission
- Process locking to prevent resource conflicts
- Automatic trial adjustment based on dataset complexity
- Model filtering based on dataset characteristics
- Comprehensive logging and progress tracking
- Both CLI and Python API interfaces

Compatibility:
- Fully compatible with improved hyperparameter_optimizer.py
- Supports all new features including pruning and enhanced logging
- Handles feature selection and stability analysis options
"""

import argparse
import fcntl
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import psutil
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_dotenv(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ.

    This avoids requiring python-dotenv and ensures AWS/MinIO credentials and
    MLFLOW_S3_ENDPOINT_URL are available to child processes launched via uv.
    """
    try:
        if not os.path.exists(env_path):
            return

        loaded = 0
        with open(env_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                # Strip optional surrounding quotes
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                loaded += 1
        if loaded:
            logger.info(f"Loaded {loaded} environment variables from {env_path}")
    except Exception as e:
        logger.warning(f"Failed to load environment from {env_path}: {e}")


class ProcessLock:
    """Ensures only one batch optimization process runs at a time."""

    def __init__(self, lock_file: str = "/tmp/batch_optimization.lock"):
        self.lock_file = lock_file
        self.lock_fd = None

    def __enter__(self):
        """Acquire the process lock."""
        try:
            self.lock_fd = open(self.lock_file, "w")
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write process info to lock file
            self.lock_fd.write(f"PID: {os.getpid()}\n")
            self.lock_fd.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.lock_fd.flush()

            logger.info("✅ Process lock acquired - this is the only batch optimization running")
            return self

        except (IOError, OSError):
            if self.lock_fd:
                self.lock_fd.close()

            logger.error("❌ Another batch optimization process is already running!")
            logger.error(
                "Only one batch optimization can run at a time to prevent thread over-subscription."
            )
            logger.error("Please wait for the current process to complete or stop it manually.")

            # Try to show info about the running process
            try:
                with open(self.lock_file, "r") as f:
                    lock_info = f.read().strip()
                    logger.error(f"Running process info:\n{lock_info}")
            except Exception as e:
                logger.error(f"Error reading lock file: {e}")

            sys.exit(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the process lock and clean up any child processes."""
        # Terminate all child processes started by this script
        try:
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for child in children:
                logger.info(f"Terminating child process {child.pid} ({child.name()})")
                child.terminate()
            gone, still_alive = psutil.wait_procs(children, timeout=5)
            for p in still_alive:
                logger.warning(f"Child process {p.pid} did not terminate, killing it.")
                p.kill()
        except psutil.NoSuchProcess:
            logger.warning("Could not find parent process to clean up children.")
        except Exception as e:
            logger.error(f"Error during child process cleanup: {e}")

        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                os.remove(self.lock_file)
                logger.info("✅ Process lock released.")
            except Exception as e:
                logger.error(f"Failed to release process lock: {e}")


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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise


def validate_file_paths(*paths: str) -> None:
    """Validate that all file paths exist."""
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")


def expand_dataset_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand dataset configurations from YAML config into individual job configurations."""
    global_defaults = config.get("global_defaults", {})
    storage_config = config.get("storage", {})
    dataset_configs = config.get("dataset_configs", {})

    if not dataset_configs:
        raise ValueError("No dataset_configs found in configuration file")

    job_configs = []

    for dataset_name, dataset_config in dataset_configs.items():
        logger.debug(f"Processing dataset config: {dataset_name}")

        # Start with global defaults
        job_config = global_defaults.copy()

        # Add storage configuration
        job_config.update(storage_config)

        # Add dataset-specific configuration
        job_config.update(dataset_config)

        # Ensure required fields are present
        required_fields = ["train_data", "test_data", "holdout_data", "target_column"]
        missing_fields = [field for field in required_fields if field not in job_config]
        if missing_fields:
            logger.error(f"Dataset '{dataset_name}' missing required fields: {missing_fields}")
            continue

        # Validate that data files exist
        try:
            validate_file_paths(
                job_config["train_data"],
                job_config["test_data"],
                job_config["holdout_data"],
            )
        except FileNotFoundError as e:
            logger.error(f"Dataset '{dataset_name}': {e}")
            continue

        # Set dataset name for reference
        job_config["dataset_name"] = dataset_name

        # Handle tags - convert dict to list of key=value strings
        if "tags" in job_config and isinstance(job_config["tags"], dict):
            tags_list = [f"{k}={v}" for k, v in job_config["tags"].items()]
            job_config["additional_tags"] = tags_list
            del job_config["tags"]  # Remove dict form

        # Use default models/metrics if not specified
        if "models" not in job_config:
            job_config["models"] = config.get(
                "default_models", ["randomforest", "xgboost", "lightgbm", "ebm"]
            )

        if "metrics" not in job_config:
            job_config["metric_combinations"] = config.get(
                "default_metric_combinations", [["precision"], ["f1"]]
            )
        else:
            # Convert single metrics list to metric combinations format
            job_config["metric_combinations"] = (
                [job_config["metrics"]]
                if isinstance(job_config["metrics"][0], str)
                else job_config["metrics"]
            )
            del job_config["metrics"]

        job_configs.append(job_config)
        logger.debug(
            f"Generated job config for dataset '{dataset_name}': {job_config.get('dataset_suffix', 'unknown')}"
        )

    logger.info(
        f"Expanded {len(dataset_configs)} dataset configs into {len(job_configs)} job configurations"
    )
    return job_configs


class BatchOptimizer:
    """Handles batch submission of hyperparameter optimization jobs."""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.optimizer_script = "src/optimizer/hyperparameter_optimizer.py"

    def generate_combinations_from_config(
        self, job_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations from YAML-based job configurations."""

        combinations = []

        for job_config in job_configs:
            dataset_name = job_config.get("dataset_name", "unknown")
            models = job_config.get("models", ["randomforest"])
            metric_combinations = job_config.get("metric_combinations", [["precision"]])

            logger.debug(
                f"Generating combinations for dataset '{dataset_name}' with {len(models)} models and {len(metric_combinations)} metric combinations"
            )

            # Apply execution rules before generating combinations
            models = self._apply_execution_rules(models, job_config)

            for model_type in models:
                for metrics in metric_combinations:
                    # Create configuration for this specific combination
                    config = job_config.copy()
                    config["model_type"] = model_type
                    config["optimization_metrics"] = metrics

                    # Apply auto-adjust trials rule
                    config = self._apply_trial_adjustment(config, job_config)

                    # Generate names based on configuration
                    metrics_str = "_".join(metrics)
                    dataset_suffix = config.get("dataset_suffix", dataset_name)

                    # Use provided experiment name or generate one
                    if "experiment_name" not in config or not config["experiment_name"]:
                        config["experiment_name"] = f"{model_type}_{dataset_suffix}_{metrics_str}"
                    # If experiment_name is provided in config, use it as-is without modification

                    config["study_name"] = (
                        f"{model_type}_optimization_{dataset_suffix}_{metrics_str}"
                    )

                    # Ensure project name includes model type
                    if not config.get("project_name"):
                        config["project_name"] = f"{model_type}_optimization"

                    # Clean up config - remove keys that shouldn't be passed to CLI
                    # Keep dataset_name and tags as they're needed for model registration
                    keys_to_remove = [
                        "models",
                        "metric_combinations",
                        "name",
                        "description",
                    ]
                    for key in keys_to_remove:
                        config.pop(key, None)

                    combinations.append(config)
                    logger.debug(f"Created combination: {config['study_name']}")

        logger.info(
            f"Generated {len(combinations)} total combinations from {len(job_configs)} job configs"
        )
        return combinations

    def _apply_execution_rules(self, models: List[str], job_config: Dict[str, Any]) -> List[str]:
        """Apply execution rules to filter models based on dataset characteristics."""
        # Get execution rules from advanced config or job config
        execution_rules = job_config.get("execution_rules", {})

        # Handle nested advanced.execution_rules structure
        if "advanced" in job_config and "execution_rules" in job_config["advanced"]:
            execution_rules = job_config["advanced"]["execution_rules"]

        if not execution_rules:
            logger.debug("No execution rules found, returning all models")
            return models

        filtered_models = models.copy()

        # Rule: Skip EBM for large datasets
        if execution_rules.get("skip_ebm_large_datasets", False):
            max_features_for_ebm = execution_rules.get("max_features_for_ebm", 1500)

            # Try to estimate number of features from dataset path or config
            num_features = self._estimate_dataset_features(job_config)

            if num_features and num_features > max_features_for_ebm:
                if "ebm" in filtered_models:
                    filtered_models.remove("ebm")
                    logger.info(
                        f"Skipping EBM model for dataset '{job_config.get('dataset_name', 'unknown')}' "
                        f"due to high feature count ({num_features} > {max_features_for_ebm})"
                    )
            else:
                logger.debug(
                    f"EBM allowed for dataset '{job_config.get('dataset_name', 'unknown')}' "
                    f"with {num_features} features (≤ {max_features_for_ebm})"
                )

        return filtered_models

    def _estimate_dataset_features(self, job_config: Dict[str, Any]) -> Optional[int]:
        """Estimate the number of features in a dataset by loading it."""
        train_data_path = job_config.get("train_data")
        # target_column = job_config.get("target_column", "y")

        if not train_data_path:
            logger.warning("No train_data path found in job config")
            return None

        try:
            # Try to load just the first few rows to get column count
            import pandas as pd

            # Convert relative path to absolute if needed
            if not train_data_path.startswith("/"):
                train_data_path = os.path.join(os.getcwd(), train_data_path)

            df = pd.read_parquet(train_data_path, nrows=1)  # Just read first row
            num_features = df.shape[1] - 1  # Subtract target column

            logger.debug(f"Estimated {num_features} features from dataset at {train_data_path}")
            return num_features

        except Exception as e:
            logger.warning(f"Failed to estimate features from {train_data_path}: {e}")
            return None

    def _apply_trial_adjustment(
        self, config: Dict[str, Any], job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply auto-adjust trials rule based on dataset size."""
        # Get execution rules from advanced config or job config
        execution_rules = job_config.get("execution_rules", {})

        # Handle nested advanced.execution_rules structure
        if "advanced" in job_config and "execution_rules" in job_config["advanced"]:
            execution_rules = job_config["advanced"]["execution_rules"]

        if not execution_rules.get("auto_adjust_trials", False):
            return config

        num_features = self._estimate_dataset_features(job_config)
        if not num_features:
            return config

        min_trials = execution_rules.get("min_trials", 50)
        max_trials = execution_rules.get("max_trials", 300)

        # Simple heuristic: more trials for more features
        if num_features < 100:
            adjusted_trials = min_trials
        elif num_features < 500:
            adjusted_trials = min(max_trials, min_trials + 50)
        elif num_features < 1000:
            adjusted_trials = min(max_trials, min_trials + 100)
        else:
            adjusted_trials = max_trials

        # Don't reduce below what was specified
        original_trials = config.get("n_trials", min_trials)
        adjusted_trials = max(adjusted_trials, original_trials)

        if adjusted_trials != original_trials:
            config["n_trials"] = adjusted_trials
            logger.info(
                f"Auto-adjusted trials for {config.get('model_type', 'unknown')} model "
                f"from {original_trials} to {adjusted_trials} (features: {num_features})"
            )

        return config

    def generate_combinations(
        self,
        model_types: List[str],
        metric_combinations: List[List[str]],
        custom_configs: Dict[str, Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for batch optimization (legacy CLI mode)."""

        combinations = []

        for model_type in model_types:
            for metrics in metric_combinations:
                # Base configuration
                config = self.base_config.copy()
                config["model_type"] = model_type
                config["optimization_metrics"] = metrics

                # Generate names based on configuration (fallback if experiment_name not provided)
                metrics_str = "_".join(metrics)

                # Use provided experiment name or generate one
                if "experiment_name" not in config or not config["experiment_name"]:
                    config["experiment_name"] = (
                        f"{model_type}_{config.get('dataset_suffix', 'training_dataset')}_{metrics_str}"
                    )

                config["study_name"] = (
                    f"{model_type}_optimization_{config.get('dataset_suffix', 'training_dataset')}_{metrics_str}"
                )
                config["project_name"] = f"{model_type}_optimization"

                # Apply custom configurations if provided
                if custom_configs and model_type in custom_configs:
                    config.update(custom_configs[model_type])

                combinations.append(config)

        return combinations

    def build_command(self, config: Dict[str, Any]) -> List[str]:
        """
        Build the command line arguments for hyperparameter_optimizer.py.

        Fully compatible with the improved hyperparameter_optimizer.py including:
        - Pruning and advanced sampling (handled automatically)
        - Feature selection via correlation analysis and file lists
        - Enhanced logging and debugging options
        - Comprehensive tagging and metadata tracking
        """

        cmd = ["uv", "run", self.optimizer_script]

        # Required arguments
        cmd.extend(["--train-data", config["train_data"]])
        cmd.extend(["--test-data", config["test_data"]])
        cmd.extend(["--holdout-data", config["holdout_data"]])
        cmd.extend(["--target-column", config["target_column"]])
        cmd.extend(["--model-type", config["model_type"]])
        cmd.extend(["--optimization-metrics"] + config["optimization_metrics"])
        cmd.extend(["--optimization-direction", config["optimization_direction"]])
        cmd.extend(["--study-name", config["study_name"]])
        cmd.extend(["--experiment-name", config["experiment_name"]])

        # Optional arguments with defaults
        cmd.extend(["--n-trials", str(config.get("n_trials", 100))])
        cmd.extend(["--random-state", str(config.get("random_state", 99))])
        cmd.extend(["--dataset-version", config.get("dataset_version", "v1.0")])
        cmd.extend(["--dataset-suffix", config.get("dataset_suffix", "training_dataset")])
        cmd.extend(
            [
                "--project-name",
                config.get("project_name", "ML Hyperparameter Optimization"),
            ]
        )

        # Add dataset name for model registration
        if config.get("dataset_name"):
            cmd.extend(["--dataset-name", config["dataset_name"]])

        # Optional arguments
        if config.get("job_timeout"):
            cmd.extend(["--timeout", str(config["job_timeout"])])
        if config.get("trial_timeout"):
            cmd.extend(["--trial-timeout", str(config["trial_timeout"])])
        if config.get("storage_url"):
            cmd.extend(["--storage-url", config["storage_url"]])
        if config.get("mlflow_uri"):
            cmd.extend(["--mlflow-uri", config["mlflow_uri"]])

        # Feature selection arguments
        if config.get("analyze_correlations"):
            cmd.append("--analyze-correlations")
            cmd.extend(
                [
                    "--correlation-threshold",
                    str(config.get("correlation_threshold", 0.95)),
                ]
            )
        elif config.get("drop_features_file"):
            cmd.extend(["--drop-features-file", config["drop_features_file"]])

        # Tags - construct tags from configuration parameters
        tags = []

        # Add dataset-specific tags from the configuration
        if config.get("tags"):
            for key, value in config["tags"].items():
                tags.append(f"{key}={value}")

        # Add CLI-related tags
        if config.get("analyze_correlations"):
            tags.append("analyze_correlations=true")
            tags.append(f"correlation_threshold={config.get('correlation_threshold', 0.95)}")

        if config.get("drop_features_file"):
            tags.append(f"drop_features_file={config['drop_features_file']}")

        # Add model and optimization tags
        tags.append(f"metrics={','.join(config['optimization_metrics'])}")
        tags.append(f"model={config['model_type']}")
        tags.append(f"optimization_direction={config['optimization_direction']}")

        # Add any additional tags from config
        if config.get("additional_tags"):
            tags.extend(config["additional_tags"])

        if tags:
            cmd.extend(["--tags"] + tags)

        # Thread configuration
        # Add threads if specified (default to 1 if not in config)
        threads = config.get("threads", 1)
        cmd.extend(["--threads", str(threads)])

        # Top-n models configuration
        top_n = config.get("top_n", 1)
        cmd.extend(["--top-n", str(top_n)])

        # Debug flag
        if config.get("debug"):
            cmd.append("--debug")

        # Light artifacts flag
        if config.get("light_artifacts"):
            cmd.append("--light-artifacts")

        return cmd

    def run_optimization(self, config: Dict[str, Any], dry_run: bool = False) -> bool:
        """Run a single optimization job."""

        cmd = self.build_command(config)

        logger.info(
            f"Starting optimization: {config['model_type']} with metrics {config['optimization_metrics']}"
        )
        logger.info(f"Experiment name: {config['experiment_name']}")
        logger.info(f"Study name: {config['study_name']}")
        logger.debug(f"Full command: {' '.join(cmd)}")

        if dry_run:
            logger.info("DRY RUN - Command would be:")
            logger.info(" ".join(cmd))
            return True

        try:
            # Run the optimization
            logger.debug(
                f"Executing subprocess with timeout: {config.get('job_timeout', 86400)} seconds"
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get("job_timeout", 86400),  # 24 hour default timeout per job
            )

            logger.debug(f"Subprocess completed with return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Subprocess stdout: {result.stdout[:500]}...")  # First 500 chars
            if result.stderr:
                logger.debug(f"Subprocess stderr: {result.stderr[:500]}...")  # First 500 chars

            if result.returncode == 0:
                logger.info(f"✅ SUCCESS: {config['study_name']}")
                return True
            else:
                logger.error(f"❌ FAILED: {config['study_name']}")
                logger.error(f"Error output: {result.stderr}")
                if result.stdout:
                    logger.error(f"Standard output: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ TIMEOUT: {config['study_name']} exceeded job timeout")
            return False
        except Exception as e:
            logger.error(f"💥 ERROR: {config['study_name']} - {e}")
            logger.debug("Exception details:", exc_info=True)
            return False

    def run_batch(
        self,
        combinations: List[Dict[str, Any]],
        dry_run: bool = False,
        delay_between_jobs: int = 30,
    ) -> Dict[str, List[str]]:
        """Run batch optimization jobs sequentially (one at a time)."""

        results = {"success": [], "failed": []}

        logger.info("=" * 80)
        logger.info("🚀 STARTING SEQUENTIAL BATCH OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Total combinations to process: {len(combinations)}")
        logger.info(f"Execution mode: {'DRY RUN' if dry_run else 'LIVE'}")

        # Show thread configuration for first combination to confirm settings
        if combinations:
            threads = combinations[0].get("threads", 1)
            logger.info(f"🧵 Thread configuration: {threads} threads per optimization job")
            logger.info(
                "⚠️  SEQUENTIAL EXECUTION: Only 1 hyperparameter optimizer will run at a time"
            )
            logger.info("💡 This prevents thread over-subscription and maintains system stability")

        logger.info("=" * 80)

        start_time = time.time()

        for i, config in enumerate(combinations, 1):
            job_start_time = time.time()

            logger.info("")
            logger.info(f"📋 [{i}/{len(combinations)}] PROCESSING JOB {i}")
            logger.info(f"   Model: {config['model_type']}")
            logger.info(f"   Metrics: {config['optimization_metrics']}")
            logger.info(f"   Experiment: {config['experiment_name']}")
            logger.info(f"   Study: {config['study_name']}")
            logger.info(f"   Threads: {config.get('threads', 1)}")

            success = self.run_optimization(config, dry_run)

            job_duration = time.time() - job_start_time

            if success:
                results["success"].append(config["study_name"])
                logger.info(
                    f"✅ [{i}/{len(combinations)}] COMPLETED in {job_duration:.1f}s: {config['study_name']}"
                )
            else:
                results["failed"].append(config["study_name"])
                logger.error(
                    f"❌ [{i}/{len(combinations)}] FAILED after {job_duration:.1f}s: {config['study_name']}"
                )

            # Progress summary
            remaining = len(combinations) - i
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / i
                estimated_remaining = avg_time * remaining
                logger.info(f"📊 Progress: {i}/{len(combinations)} jobs, {remaining} remaining")
                logger.info(
                    f"⏱️  Elapsed: {elapsed:.1f}s, Est. remaining: {estimated_remaining:.1f}s"
                )

            # Add delay between jobs (except for dry runs and last job)
            if not dry_run and i < len(combinations) and delay_between_jobs > 0:
                logger.info(
                    f"⏸️  Waiting {delay_between_jobs}s before next job (prevents resource conflicts)..."
                )
                time.sleep(delay_between_jobs)

        total_duration = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🏁 BATCH OPTIMIZATION COMPLETED in {total_duration:.1f}s")
        logger.info("=" * 80)

        return results


def create_default_configurations():
    """Create default configuration combinations for comprehensive optimization."""

    # Model types to test
    model_types = ["randomforest", "xgboost", "lightgbm", "ebm"]

    # Metric combinations - single metrics and important combinations
    metric_combinations = [
        ["precision"],
        #    ["recall"],
        ["f1"],
        ["roc_auc"],
        ["pr_auc"],
        #    ["precision", "recall"],
        ["precision", "f1"],
        ["f1", "roc_auc"],
        #    ["precision", "recall", "f1"],
        ["roc_auc", "pr_auc"],
    ]

    # Custom configurations per model type
    custom_configs = {
        "randomforest": {
            "n_trials": 150,  # More trials for RF due to more hyperparameters
        },
        "xgboost": {
            "n_trials": 200,  # More trials for XGB
        },
        "lightgbm": {
            "n_trials": 200,  # More trials for LGBM
        },
        "ebm": {
            "n_trials": 100,  # Fewer trials for EBM (slower)
        },
    }

    return model_types, metric_combinations, custom_configs


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file. If provided, most other arguments will be ignored.",
    )

    # Dataset arguments (required if no config file)
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training dataset (parquet format) - required if no config file",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset (parquet format) - required if no config file",
    )
    parser.add_argument(
        "--holdout-data",
        type=str,
        help="Path to holdout dataset (parquet format) - required if no config file",
    )
    parser.add_argument("--target-column", default="y", type=str, help="Name of the target column")

    # Batch configuration
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["randomforest", "xgboost", "lightgbm", "ebm"],
        help="Model types to optimize (default: all)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["precision", "recall", "f1", "roc_auc", "pr_auc"],
        help="Single metric to optimize (overrides default combinations)",
    )
    parser.add_argument(
        "--optimization-direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="Direction of optimization",
    )

    # Job configuration
    parser.add_argument(
        "--n-trials",
        default=100,
        type=int,
        help="Default number of trials per optimization",
    )
    parser.add_argument("--timeout", type=int, help="Timeout per optimization in seconds")
    parser.add_argument("--job-timeout", type=int, help="Timeout per job in seconds")
    parser.add_argument("--trial-timeout", type=int, help="Timeout per individual trial in seconds")
    parser.add_argument(
        "--delay-between-jobs",
        default=30,
        type=int,
        help="Delay between jobs in seconds",
    )
    parser.add_argument(
        "--random-state", default=99, type=int, help="Random state for reproducibility"
    )

    # Storage configuration
    parser.add_argument("--storage-url", type=str, help="Database URL for Optuna storage")
    parser.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--dataset-version", help="Dataset version tag")
    parser.add_argument(
        "--dataset-suffix",
        type=str,
        help="Suffix for naming studies and experiments - required if no config file",
    )

    # MLflow configuration
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLflow experiment name (if not provided, will be auto-generated per combination)",
    )
    parser.add_argument(
        "--project-name",
        default="ML Hyperparameter Optimization",
        help="Project name for MLflow tags",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would be run without executing",
    )

    # Feature selection arguments
    parser.add_argument(
        "--analyze-correlations",
        action="store_true",
        help="Analyze and drop highly correlated features for all jobs in this batch.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold to use when --analyze-correlations is set.",
    )
    parser.add_argument(
        "--drop-features-file",
        type=str,
        help="Path to a file with features to drop for all jobs in this batch.",
    )

    # Tags arguments
    parser.add_argument(
        "--tags",
        nargs="+",
        type=str,
        help="Additional tags in key=value format for all jobs (e.g., --tags batch_id=001 experiment_type=test)",
    )

    # Thread configuration
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads to use for model training and optimization (default: from config or 1)",
    )

    # Top models training
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top models to train and evaluate on holdout data (default: from config or 1)",
    )

    # Debug arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed troubleshooting",
    )

    return parser


def run_batch_optimization(
    config_path: Optional[str] = None,
    train_data: Optional[str] = None,
    test_data: Optional[str] = None,
    holdout_data: Optional[str] = None,
    models: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    n_trials: int = 100,
    dataset_suffix: Optional[str] = None,
    storage_url: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    dry_run: bool = False,
    **kwargs,
) -> Dict[str, List[str]]:
    """
    Run batch hyperparameter optimization programmatically.

    This function provides a Python API for batch optimization, allowing you to
    submit multiple hyperparameter optimization jobs with different model types
    and metric combinations.

    Args:
        config_path: Path to YAML configuration file. If provided, most other args ignored
        train_data: Path to training dataset (parquet format)
        test_data: Path to test dataset (parquet format)
        holdout_data: Path to holdout dataset (parquet format)
        models: List of model types to optimize ('randomforest', 'xgboost', 'lightgbm', 'ebm')
        metrics: List of metrics to optimize ('precision', 'recall', 'f1', 'roc_auc', 'pr_auc')
        n_trials: Number of trials per optimization
        dataset_suffix: Suffix for naming studies and experiments
        storage_url: Database URL for Optuna storage
        mlflow_uri: MLflow tracking URI
        dry_run: If True, show commands without executing
        **kwargs: Additional parameters (threads, delay_between_jobs, etc.)

    Returns:
        Dict with 'successful' and 'failed' lists of job descriptions

    Raises:
        FileNotFoundError: If required data files don't exist
        ValueError: If configuration is invalid
        RuntimeError: If batch optimization fails

    Example:
        >>> from src.optimizer.batch_optimization import run_batch_optimization
        >>>
        >>> # Using configuration file
        >>> results = run_batch_optimization(config_path="batch_config.yaml")
        >>>
        >>> # Direct parameters
        >>> results = run_batch_optimization(
        ...     train_data="train.parquet",
        ...     test_data="test.parquet",
        ...     holdout_data="holdout.parquet",
        ...     models=["randomforest", "xgboost"],
        ...     metrics=["precision", "f1"],
        ...     n_trials=50,
        ...     dataset_suffix="experiment_1"
        ... )
        >>> print(f"Successful jobs: {len(results['successful'])}")
    """
    try:
        # Ensure env vars from .env (AWS creds, MLFLOW_S3_ENDPOINT_URL) are loaded
        _load_dotenv()
        # Validate input parameters if not using config file
        if not config_path and train_data and test_data and holdout_data:
            validate_file_paths(train_data, test_data, holdout_data)

        with ProcessLock():
            # Build args similar to command line
            class Args:
                def __init__(self):
                    self.config = config_path
                    self.train_data = train_data
                    self.test_data = test_data
                    self.holdout_data = holdout_data
                    self.models = models
                    self.metrics = metrics
                    self.n_trials = n_trials
                    self.dataset_suffix = dataset_suffix
                    self.storage_url = storage_url
                    self.mlflow_uri = mlflow_uri
                    self.dry_run = dry_run
                    self.debug = kwargs.get("debug", False)

                    # Set other defaults
                    self.target_column = kwargs.get("target_column", "y")
                    self.optimization_direction = kwargs.get("optimization_direction", "maximize")
                    self.timeout = kwargs.get("timeout")
                    self.job_timeout = kwargs.get("job_timeout", 86400)
                    self.delay_between_jobs = kwargs.get("delay_between_jobs", 30)
                    self.random_state = kwargs.get("random_state", 99)
                    self.dataset_version = kwargs.get("dataset_version")
                    self.experiment_name = kwargs.get("experiment_name")
                    self.project_name = kwargs.get("project_name", "ML Hyperparameter Optimization")
                    self.analyze_correlations = kwargs.get("analyze_correlations", False)
                    self.correlation_threshold = kwargs.get("correlation_threshold", 0.95)
                    self.drop_features_file = kwargs.get("drop_features_file")
                    self.tags = kwargs.get("tags")
                    self.threads = kwargs.get("threads")
                    self.top_n = kwargs.get("top_n")

            args = Args()

            # Configure logging
            if args.debug:
                logging.getLogger().setLevel(logging.DEBUG)
                logger.debug("Debug logging enabled for batch optimization")

            logger.debug(f"API batch optimization arguments: {vars(args)}")

            return _execute_batch_optimization(args)

    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        raise


def main(args=None) -> int:
    """
    Main execution function for CLI.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    if not args:
        args = parser.parse_args()

    # Configure logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled for batch optimization")

    logger.debug(f"Batch optimization arguments: {vars(args)}")

    try:
        # Load .env so subprocesses inherit MinIO/MLflow credentials
        _load_dotenv()
        # Use process lock to ensure only one batch optimization runs at a time
        # This prevents thread over-subscription and resource conflicts
        with ProcessLock():
            _execute_batch_optimization(args)
        return 0
    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        return 1


def _execute_batch_optimization(args):
    """Execute the batch optimization with process lock acquired."""
    # Handle configuration modes
    if args.config:
        # Config file mode
        logger.info(f"Using configuration file: {args.config}")

        # Load and expand configuration
        config = load_config(args.config)
        job_configs = expand_dataset_configs(config)

        # Override with command line arguments if provided
        for job_config in job_configs:
            # Command line overrides for execution options
            if args.dry_run:
                job_config["dry_run"] = True
            if args.debug:
                job_config["debug"] = True
            if args.threads is not None:
                job_config["threads"] = args.threads
            if getattr(args, "top_n", None) is not None:
                job_config["top_n"] = args.top_n
            if args.delay_between_jobs is not None:
                job_config["delay_between_jobs"] = args.delay_between_jobs
            if args.job_timeout is not None:
                job_config["job_timeout"] = args.job_timeout
            if getattr(args, "trial_timeout", None) is not None:
                job_config["trial_timeout"] = args.trial_timeout

        # Create batch optimizer and generate combinations
        batch_optimizer = BatchOptimizer({})  # Empty base config since we're using job configs
        combinations = batch_optimizer.generate_combinations_from_config(job_configs)

    else:
        # Command line mode (original behavior)
        logger.info("Using command line arguments")

        # Validate required arguments for CLI mode
        required_args = ["train_data", "test_data", "holdout_data", "dataset_suffix"]
        missing_args = [arg for arg in required_args if not getattr(args, arg)]
        if missing_args:
            logger.error(f"Missing required arguments for CLI mode: {missing_args}")
            logger.error("Either provide --config or all required dataset arguments")
            sys.exit(1)

        # Validate that data files exist
        try:
            validate_file_paths(args.train_data, args.test_data, args.holdout_data)
        except FileNotFoundError as e:
            logger.error(f"CLI mode validation failed: {e}")
            sys.exit(1)

        # Parse additional tags
        additional_tags = parse_tags(args.tags) if args.tags else {}
        additional_tags_list = [f"{k}={v}" for k, v in additional_tags.items()]

        # Base configuration from arguments
        base_config = {
            "train_data": args.train_data,
            "test_data": args.test_data,
            "holdout_data": args.holdout_data,
            "target_column": args.target_column,
            "optimization_direction": args.optimization_direction,
            "n_trials": args.n_trials,
            "random_state": args.random_state,
            "dataset_version": args.dataset_version,
            "dataset_suffix": args.dataset_suffix,
            "project_name": args.project_name,
        }

        # Add experiment name if provided (will be used for all combinations)
        if args.experiment_name:
            base_config["experiment_name"] = args.experiment_name

        # Add optional configurations
        if args.timeout:
            base_config["timeout"] = args.timeout
        if args.storage_url:
            base_config["storage_url"] = args.storage_url
        if args.mlflow_uri:
            base_config["mlflow_uri"] = args.mlflow_uri
        if args.job_timeout:
            base_config["job_timeout"] = args.job_timeout
        if getattr(args, "trial_timeout", None):
            base_config["trial_timeout"] = args.trial_timeout

        # Add feature selection configurations
        if args.analyze_correlations:
            base_config["analyze_correlations"] = True
            base_config["correlation_threshold"] = args.correlation_threshold
        elif args.drop_features_file:
            base_config["drop_features_file"] = args.drop_features_file

        # Add debug flag
        if args.debug:
            base_config["debug"] = True

        # Add threads configuration
        if args.threads is not None:
            base_config["threads"] = args.threads

        # Add top-n configuration
        if getattr(args, "top_n", None) is not None:
            base_config["top_n"] = args.top_n

        # Add additional tags
        if additional_tags_list:
            base_config["additional_tags"] = additional_tags_list

        logger.debug(f"Base configuration: {base_config}")

        # Determine model types and metric combinations
        if args.models:
            model_types = args.models
        else:
            model_types, _, _ = create_default_configurations()

        if args.metrics:
            # Single metric combination provided
            metric_combinations = [args.metrics]
            custom_configs = {}
        else:
            # Use default combinations
            _, metric_combinations, custom_configs = create_default_configurations()

        # Create batch optimizer and generate combinations (legacy method)
        batch_optimizer = BatchOptimizer(base_config)
        combinations = batch_optimizer.generate_combinations(
            model_types, metric_combinations, custom_configs
        )

    logger.info(f"Generated {len(combinations)} optimization combinations")
    logger.debug("Generated combinations:")
    for i, combo in enumerate(combinations, 1):
        logger.debug(
            f"  {i}. {combo['model_type']} - {combo['optimization_metrics']} - Exp: {combo['experiment_name']} - Study: {combo['study_name']}"
        )

    # Run batch optimization
    results = batch_optimizer.run_batch(
        combinations, dry_run=args.dry_run, delay_between_jobs=args.delay_between_jobs
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH OPTIMIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total jobs: {len(combinations)}")
    logger.info(f"Successful: {len(results['success'])}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results["success"]:
        logger.info("\nSuccessful optimizations:")
        for study in results["success"]:
            logger.info(f"  ✅ {study}")

    if results["failed"]:
        logger.info("\nFailed optimizations:")
        for study in results["failed"]:
            logger.info(f"  ❌ {study}")

    # Exit with appropriate code
    if results["failed"] and not args.dry_run:
        logger.error("Some optimizations failed!")
        sys.exit(1)
    else:
        logger.info("Batch optimization completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
