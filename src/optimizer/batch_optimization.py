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
import yaml
from itertools import product
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_tags(tags_list: List[str]) -> Dict[str, str]:
    """Parse key=value tag pairs into a dictionary."""
    tags = {}
    if tags_list:
        for tag in tags_list:
            if '=' in tag:
                key, value = tag.split('=', 1)  # Split on first '=' only
                tags[key.strip()] = value.strip()
            else:
                logger.warning(f"Invalid tag format: '{tag}'. Expected 'key=value' format.")
    return tags


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
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


def expand_dataset_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand dataset configurations from YAML config into individual job configurations."""
    global_defaults = config.get('global_defaults', {})
    storage_config = config.get('storage', {})
    dataset_configs = config.get('dataset_configs', {})
    
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
        required_fields = ['train_data', 'test_data', 'holdout_data', 'target_column']
        missing_fields = [field for field in required_fields if field not in job_config]
        if missing_fields:
            logger.error(f"Dataset '{dataset_name}' missing required fields: {missing_fields}")
            continue
        
        # Set dataset name for reference
        job_config['dataset_name'] = dataset_name
        
        # Handle tags - convert dict to list of key=value strings
        if 'tags' in job_config and isinstance(job_config['tags'], dict):
            tags_list = [f"{k}={v}" for k, v in job_config['tags'].items()]
            job_config['additional_tags'] = tags_list
            del job_config['tags']  # Remove dict form
        
        # Use default models/metrics if not specified
        if 'models' not in job_config:
            job_config['models'] = config.get('default_models', ["randomforest", "xgboost", "lightgbm", "ebm"])
        
        if 'metrics' not in job_config:
            job_config['metric_combinations'] = config.get('default_metric_combinations', [["precision"], ["f1"]])
        else:
            # Convert single metrics list to metric combinations format
            job_config['metric_combinations'] = [job_config['metrics']] if isinstance(job_config['metrics'][0], str) else job_config['metrics']
            del job_config['metrics']
        
        job_configs.append(job_config)
        logger.debug(f"Generated job config for dataset '{dataset_name}': {job_config.get('dataset_suffix', 'unknown')}")
    
    logger.info(f"Expanded {len(dataset_configs)} dataset configs into {len(job_configs)} job configurations")
    return job_configs


class BatchOptimizer:
    """Handles batch submission of hyperparameter optimization jobs."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.optimizer_script = "src/optimizer/hyperparameter_optimizer.py"
        
    def generate_combinations_from_config(self, job_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations from YAML-based job configurations."""
        
        combinations = []
        
        for job_config in job_configs:
            dataset_name = job_config.get('dataset_name', 'unknown')
            models = job_config.get('models', ["randomforest"])
            metric_combinations = job_config.get('metric_combinations', [["precision"]])
            
            logger.debug(f"Generating combinations for dataset '{dataset_name}' with {len(models)} models and {len(metric_combinations)} metric combinations")
            
            for model_type in models:
                for metrics in metric_combinations:
                    # Create configuration for this specific combination
                    config = job_config.copy()
                    config["model_type"] = model_type
                    config["optimization_metrics"] = metrics
                    
                    # Generate names based on configuration
                    metrics_str = "_".join(metrics)
                    dataset_suffix = config.get('dataset_suffix', dataset_name)
                    
                    # Use provided experiment name or generate one
                    if "experiment_name" not in config or not config["experiment_name"]:
                        config["experiment_name"] = f"{model_type}_{dataset_suffix}_{metrics_str}"
                    else:
                        # Append model and metrics to provided experiment name for uniqueness
                        base_exp_name = config["experiment_name"]
                        config["experiment_name"] = f"{base_exp_name}_{model_type}_{metrics_str}"
                    
                    config["study_name"] = f"{model_type}_optimization_{dataset_suffix}_{metrics_str}"
                    
                    # Ensure project name includes model type
                    if not config.get("project_name"):
                        config["project_name"] = f"{model_type}_optimization"
                    
                    # Clean up config - remove keys that shouldn't be passed to CLI
                    keys_to_remove = ['models', 'metric_combinations', 'dataset_name', 'name', 'description']
                    for key in keys_to_remove:
                        config.pop(key, None)
                    
                    combinations.append(config)
                    logger.debug(f"Created combination: {config['study_name']}")
        
        logger.info(f"Generated {len(combinations)} total combinations from {len(job_configs)} job configs")
        return combinations
    
    def generate_combinations(self, 
                            model_types: List[str],
                            metric_combinations: List[List[str]],
                            custom_configs: Dict[str, Dict] = None) -> List[Dict[str, Any]]:
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
                    config["experiment_name"] = f"{model_type}_{config.get('dataset_suffix', 'training_dataset')}_{metrics_str}"
                
                config["study_name"] = f"{model_type}_optimization_{config.get('dataset_suffix', 'training_dataset')}_{metrics_str}"
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
        
        # Optional arguments with defaults
        cmd.extend(["--n-trials", str(config.get("n_trials", 100))])
        cmd.extend(["--random-state", str(config.get("random_state", 99))])
        cmd.extend(["--dataset-version", config.get("dataset_version", "v1.0")])
        cmd.extend(["--dataset-suffix", config.get("dataset_suffix", "training_dataset")])
        cmd.extend(["--project-name", config.get("project_name", "ML Hyperparameter Optimization")])
        
        # Optional arguments
        if config.get("timeout"):
            cmd.extend(["--timeout", str(config["timeout"])])
        if config.get("storage_url"):
            cmd.extend(["--storage-url", config["storage_url"]])
        if config.get("mlflow_uri"):
            cmd.extend(["--mlflow-uri", config["mlflow_uri"]])
        
        # Feature selection arguments
        if config.get("analyze_correlations"):
            cmd.append("--analyze-correlations")
            cmd.extend(["--correlation-threshold", str(config.get("correlation_threshold", 0.95))])
        elif config.get("drop_features_file"):
            cmd.extend(["--drop-features-file", config["drop_features_file"]])
        
        # Tags - construct tags from configuration parameters
        tags = []
        
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
        
        # Debug flag
        if config.get("debug"):
            cmd.append("--debug")
            
        return cmd
    
    def run_optimization(self, config: Dict[str, Any], dry_run: bool = False) -> bool:
        """Run a single optimization job."""
        
        cmd = self.build_command(config)
        
        logger.info(f"Starting optimization: {config['model_type']} with metrics {config['optimization_metrics']}")
        logger.info(f"Experiment name: {config['experiment_name']}")
        logger.info(f"Study name: {config['study_name']}")
        logger.debug(f"Full command: {' '.join(cmd)}")
        
        if dry_run:
            logger.info("DRY RUN - Command would be:")
            logger.info(" ".join(cmd))
            return True
        
        try:
            # Run the optimization
            logger.debug(f"Executing subprocess with timeout: {config.get('job_timeout', 3600)} seconds")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get("job_timeout", 3600)  # 1 hour default timeout per job
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
            logger.debug(f"Exception details:", exc_info=True)
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
    
    # Configuration file
    parser.add_argument("--config", type=str,
                       help="Path to YAML configuration file. If provided, most other arguments will be ignored.")
    
    # Dataset arguments (required if no config file)
    parser.add_argument("--train-data", type=str,
                       help="Path to training dataset (parquet format) - required if no config file")
    parser.add_argument("--test-data", type=str,
                       help="Path to test dataset (parquet format) - required if no config file")
    parser.add_argument("--holdout-data", type=str,
                       help="Path to holdout dataset (parquet format) - required if no config file")
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
    parser.add_argument("--dataset-suffix", type=str,
                       help="Suffix for naming studies and experiments - required if no config file")
    
    # MLflow configuration
    parser.add_argument("--experiment-name", type=str,
                       help="MLflow experiment name (if not provided, will be auto-generated per combination)")
    parser.add_argument("--project-name", default="ML Hyperparameter Optimization",
                       help="Project name for MLflow tags")
    
    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show commands that would be run without executing")
    parser.add_argument("--parallel", action="store_true",
                       help="Run optimizations in parallel (not implemented)")
    
    # Feature selection arguments
    parser.add_argument("--analyze-correlations", action="store_true",
                        help="Analyze and drop highly correlated features for all jobs in this batch.")
    parser.add_argument("--correlation-threshold", type=float, default=0.95,
                        help="Correlation threshold to use when --analyze-correlations is set.")
    parser.add_argument("--drop-features-file", type=str,
                        help="Path to a file with features to drop for all jobs in this batch.")
    
    # Tags arguments
    parser.add_argument("--tags", nargs="+", type=str,
                       help="Additional tags in key=value format for all jobs (e.g., --tags batch_id=001 experiment_type=test)")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for detailed troubleshooting")

    return parser


def main(args=None):
    """Main execution function."""
    parser = create_argument_parser()
    if not args:
        args = parser.parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled for batch optimization")
    
    logger.debug(f"Batch optimization arguments: {vars(args)}")
    
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
                job_config['dry_run'] = True
            if args.debug:
                job_config['debug'] = True
            if args.parallel:
                job_config['parallel'] = True
            if args.delay_between_jobs is not None:
                job_config['delay_between_jobs'] = args.delay_between_jobs
            if args.job_timeout is not None:
                job_config['job_timeout'] = args.job_timeout
        
        # Create batch optimizer and generate combinations
        batch_optimizer = BatchOptimizer({})  # Empty base config since we're using job configs
        combinations = batch_optimizer.generate_combinations_from_config(job_configs)
        
    else:
        # Command line mode (original behavior)
        logger.info("Using command line arguments")
        
        # Validate required arguments for CLI mode
        required_args = ['train_data', 'test_data', 'holdout_data', 'dataset_suffix']
        missing_args = [arg for arg in required_args if not getattr(args, arg)]
        if missing_args:
            logger.error(f"Missing required arguments for CLI mode: {missing_args}")
            logger.error("Either provide --config or all required dataset arguments")
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
        
        # Add feature selection configurations
        if args.analyze_correlations:
            base_config["analyze_correlations"] = True
            base_config["correlation_threshold"] = args.correlation_threshold
        elif args.drop_features_file:
            base_config["drop_features_file"] = args.drop_features_file
        
        # Add debug flag
        if args.debug:
            base_config["debug"] = True
        
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
        logger.debug(f"  {i}. {combo['model_type']} - {combo['optimization_metrics']} - Exp: {combo['experiment_name']} - Study: {combo['study_name']}")
    
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
