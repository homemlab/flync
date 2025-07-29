#!/usr/bin/env python3
"""
Batch Hyperparameter Optimization Script

Submits multiple hyperparameter optimization jobs with different combinations of:
- Model types (RandomForest, XGBoost, LightGBM, EBM)
- Optimization metrics (precision, recall, f1, roc_auc, pr_auc)
- Various parameter combinations

Based on the hyperparameter_optimizer.py script.
"""

import argparse
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class BatchOptimizer:
    """Handles batch submission of hyperparameter optimization jobs."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.optimizer_script = "src/optimizer/hyperparameter_optimizer.py"
        
    def generate_combinations(self, 
                            model_types: List[str],
                            metric_combinations: List[List[str]],
                            custom_configs: Dict[str, Dict] = None) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for batch optimization."""
        
        combinations = []
        
        for model_type in model_types:
            for metrics in metric_combinations:
                # Base configuration
                config = self.base_config.copy()
                config["model_type"] = model_type
                config["optimization_metrics"] = metrics
                
                # Generate names based on configuration
                metrics_str = "_".join(metrics)
                config["study_name"] = f"{model_type}_optimization_{config.get('dataset_suffix', 'redux')}_{metrics_str}"
                config["experiment_name"] = f"{model_type}_{config.get('dataset_suffix', 'redux')}_{metrics_str}"
                config["project_name"] = f"{model_type}_optimization"
                
                # Apply custom configurations if provided
                if custom_configs and model_type in custom_configs:
                    config.update(custom_configs[model_type])
                
                combinations.append(config)
        
        return combinations
    
    def build_command(self, config: Dict[str, Any]) -> List[str]:
        """Build the command line arguments for hyperparameter_optimizer.py."""
        
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
        cmd.extend(["--project-name", config["project_name"]])
        
        # Optional arguments with defaults
        cmd.extend(["--n-trials", str(config.get("n_trials", 100))])
        cmd.extend(["--random-state", str(config.get("random_state", 99))])
        cmd.extend(["--dataset-version", config.get("dataset_version", "redux_no-names_no-cpat")])
        
        # Optional arguments
        if config.get("timeout"):
            cmd.extend(["--timeout", str(config["timeout"])])
        if config.get("storage_url"):
            cmd.extend(["--storage-url", config["storage_url"]])
        if config.get("mlflow_uri"):
            cmd.extend(["--mlflow-uri", config["mlflow_uri"]])
            
        return cmd
    
    def run_optimization(self, config: Dict[str, Any], dry_run: bool = False) -> bool:
        """Run a single optimization job."""
        
        cmd = self.build_command(config)
        
        logger.info(f"Starting optimization: {config['model_type']} with metrics {config['optimization_metrics']}")
        logger.info(f"Study name: {config['study_name']}")
        
        if dry_run:
            logger.info("DRY RUN - Command would be:")
            logger.info(" ".join(cmd))
            return True
        
        try:
            # Run the optimization
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get("job_timeout", 3600)  # 1 hour default timeout per job
            )
            
            if result.returncode == 0:
                logger.info(f"✅ SUCCESS: {config['study_name']}")
                return True
            else:
                logger.error(f"❌ FAILED: {config['study_name']}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ TIMEOUT: {config['study_name']} exceeded job timeout")
            return False
        except Exception as e:
            logger.error(f"💥 ERROR: {config['study_name']} - {e}")
            return False
    
    def run_batch(self, combinations: List[Dict[str, Any]], 
                  dry_run: bool = False, 
                  parallel: bool = False,
                  delay_between_jobs: int = 30) -> Dict[str, List[str]]:
        """Run batch optimization jobs."""
        
        results = {"success": [], "failed": []}
        
        logger.info(f"Starting batch optimization with {len(combinations)} combinations")
        logger.info(f"Dry run: {dry_run}, Parallel: {parallel}")
        
        if parallel:
            logger.warning("Parallel execution not implemented yet. Running sequentially.")
        
        for i, config in enumerate(combinations, 1):
            logger.info(f"[{i}/{len(combinations)}] Running configuration...")
            
            success = self.run_optimization(config, dry_run)
            
            if success:
                results["success"].append(config["study_name"])
            else:
                results["failed"].append(config["study_name"])
            
            # Add delay between jobs (except for dry runs and last job)
            if not dry_run and i < len(combinations) and delay_between_jobs > 0:
                logger.info(f"Waiting {delay_between_jobs} seconds before next job...")
                time.sleep(delay_between_jobs)
        
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
        ["roc_auc", "pr_auc"]
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
        }
    }
    
    return model_types, metric_combinations, custom_configs


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument("--train-data", required=True, type=str,
                       help="Path to training dataset (parquet format)")
    parser.add_argument("--test-data", required=True, type=str,
                       help="Path to test dataset (parquet format)")
    parser.add_argument("--holdout-data", required=True, type=str,
                       help="Path to holdout dataset (parquet format)")
    parser.add_argument("--target-column", default="y", type=str,
                       help="Name of the target column")
    
    # Batch configuration
    parser.add_argument("--models", nargs="+", 
                       choices=["randomforest", "xgboost", "lightgbm", "ebm"],
                       help="Model types to optimize (default: all)")
    parser.add_argument("--metrics", nargs="+",
                       choices=["precision", "recall", "f1", "roc_auc", "pr_auc"],
                       help="Single metric to optimize (overrides default combinations)")
    parser.add_argument("--optimization-direction", default="maximize",
                       choices=["maximize", "minimize"],
                       help="Direction of optimization")
    
    # Job configuration
    parser.add_argument("--n-trials", default=100, type=int,
                       help="Default number of trials per optimization")
    parser.add_argument("--timeout", type=int,
                       help="Timeout per optimization in seconds")
    parser.add_argument("--job-timeout", default=3600, type=int,
                       help="Timeout per job in seconds")
    parser.add_argument("--delay-between-jobs", default=30, type=int,
                       help="Delay between jobs in seconds")
    parser.add_argument("--random-state", default=99, type=int,
                       help="Random state for reproducibility")
    
    # Storage configuration
    parser.add_argument("--storage-url", type=str,
                       help="Database URL for Optuna storage")
    parser.add_argument("--mlflow-uri", type=str,
                       help="MLflow tracking URI")
    parser.add_argument("--dataset-version",
                       help="Dataset version tag")
    parser.add_argument("--dataset-suffix", required=True, type=str,
                       help="Suffix for naming studies and experiments")
    
    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show commands that would be run without executing")
    parser.add_argument("--parallel", action="store_true",
                       help="Run optimizations in parallel (not implemented)")
    
    return parser


def main():
    """Main execution function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
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
    }
    
    # Add optional configurations
    if args.timeout:
        base_config["timeout"] = args.timeout
    if args.storage_url:
        base_config["storage_url"] = args.storage_url
    if args.mlflow_uri:
        base_config["mlflow_uri"] = args.mlflow_uri
    if args.job_timeout:
        base_config["job_timeout"] = args.job_timeout
    
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
    
    # Create batch optimizer and generate combinations
    batch_optimizer = BatchOptimizer(base_config)
    combinations = batch_optimizer.generate_combinations(
        model_types, metric_combinations, custom_configs
    )
    
    logger.info(f"Generated {len(combinations)} optimization combinations")
    
    # Run batch optimization
    results = batch_optimizer.run_batch(
        combinations,
        dry_run=args.dry_run,
        parallel=args.parallel,
        delay_between_jobs=args.delay_between_jobs
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BATCH OPTIMIZATION SUMMARY")
    logger.info("="*80)
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
    main()
