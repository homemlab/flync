#!/usr/bin/env python3
"""
Example Usage Script

This script demonstrates how to use the improved hyperparameter optimization tool.
"""

import subprocess
import os
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def main():
    """Demonstrate the usage of the hyperparameter optimization tool."""
    
    print_section("HYPERPARAMETER OPTIMIZATION TOOL DEMO")
    
    # Check if we have the required data files
    pos_file = "../../ncr_dim_redux.parquet"
    neg_file = "../../pcg_dim_redux.parquet"
    
    if not (Path(pos_file).exists() and Path(neg_file).exists()):
        print(f"⚠️  Data files not found: {pos_file}, {neg_file}")
        print("This demo expects the original data files to be present in the parent directory.")
        print("Please ensure these files are in the sandbox directory.")
        return
    
    print_section("STEP 1: DATA PREPARATION")
    print("Preparing train/test/holdout splits from original data...")
    
    prep_cmd = [
        "python", "../prepare_data.py",
        "--positive-file", pos_file,
        "--negative-file", neg_file,
        "--output-dir", "demo_datasets",
        "--target-column", "y",
        "--test-size", "0.2",
        "--holdout-size", "0.15"
    ]
    
    print("Command:")
    print("  " + " \\\n    ".join(prep_cmd))
    
    try:
        subprocess.run(prep_cmd, check=True)
        print("✅ Data preparation completed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Data preparation failed")
        return
    
    print_section("STEP 2: HYPERPARAMETER OPTIMIZATION")
    
    # Example 1: RandomForest optimization
    print("Example 1: RandomForest optimizing for precision and F1-score")
    
    rf_cmd = [
        "python", "../hyperparameter_optimizer.py",
        "--train-data", "demo_datasets/train.parquet",
        "--test-data", "demo_datasets/test.parquet",
        "--holdout-data", "demo_datasets/holdout.parquet",
        "--target-column", "y",
        "--model-type", "randomforest",
        "--optimization-metrics", "precision", "f1",
        "--optimization-direction", "maximize",
        "--study-name", "demo_rf_precision_f1",
        "--n-trials", "20",  # Reduced for demo
        "--experiment-name", "Demo_RandomForest_Optimization",
        "--project-name", "Hyperparameter Optimization Demo",
        "--dataset-version", "demo_v1"
    ]
    
    print("\nCommand:")
    print("  " + " \\\n    ".join(rf_cmd))
    
    user_input = input("\nRun RandomForest optimization? (y/n): ")
    if user_input.lower() == 'y':
        try:
            subprocess.run(rf_cmd, check=True)
            print("✅ RandomForest optimization completed!")
        except subprocess.CalledProcessError:
            print("❌ RandomForest optimization failed")
    
    # Example 2: XGBoost optimization
    print("\n" + "-"*60)
    print("Example 2: XGBoost optimizing for recall and precision")
    
    xgb_cmd = [
        "python", "../hyperparameter_optimizer.py",
        "--train-data", "demo_datasets/train.parquet",
        "--test-data", "demo_datasets/test.parquet",
        "--holdout-data", "demo_datasets/holdout.parquet",
        "--target-column", "y",
        "--model-type", "xgboost",
        "--optimization-metrics", "recall", "precision",
        "--optimization-direction", "maximize",
        "--study-name", "demo_xgb_recall_precision",
        "--n-trials", "20",  # Reduced for demo
        "--experiment-name", "Demo_XGBoost_Optimization",
        "--project-name", "Hyperparameter Optimization Demo",
        "--dataset-version", "demo_v1"
    ]
    
    print("\nCommand:")
    print("  " + " \\\n    ".join(xgb_cmd))
    
    user_input = input("\nRun XGBoost optimization? (y/n): ")
    if user_input.lower() == 'y':
        try:
            subprocess.run(xgb_cmd, check=True)
            print("✅ XGBoost optimization completed!")
        except subprocess.CalledProcessError:
            print("❌ XGBoost optimization failed")
    
    print_section("STEP 3: VIEW RESULTS")
    
    print("To view the optimization results:")
    print("1. Start MLflow UI:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("\n2. Open your browser to: http://localhost:5000")
    print("\n3. Navigate to the experiments to see:")
    print("   - Trial comparisons")
    print("   - Model metrics")
    print("   - Feature importance plots")
    print("   - ROC and PR curves")
    print("   - Optuna optimization plots")
    
    print("\nOptuna study databases:")
    print("   - demo_rf_precision_f1")
    print("   - demo_xgb_recall_precision")
    
    print_section("KEY IMPROVEMENTS")
    
    improvements = [
        "✅ Modular, object-oriented design",
        "✅ Complete CLI interface with argparse",
        "✅ Flexible optimization metrics (multiple metrics supported)",
        "✅ Proper train/test/holdout data flow",
        "✅ Comprehensive error handling and logging",
        "✅ Enhanced feature importance analysis with percentages",
        "✅ Advanced stratification options (groups, balancing)",
        "✅ MLflow integration with visualizations",
        "✅ Optuna study persistence and resumability",
        "✅ Type hints and documentation",
        "✅ Configurable hyperparameter spaces",
        "✅ Professional code structure"
    ]
    
    for improvement in improvements:
        print(improvement)
    
    print_section("USAGE PATTERNS")
    
    print("1. Single metric optimization:")
    print("   --optimization-metrics precision")
    
    print("\n2. Multi-metric optimization:")
    print("   --optimization-metrics precision recall f1")
    
    print("\n3. Resume existing study:")
    print("   Use the same --study-name to continue optimization")
    
    print("\n4. Advanced stratification:")
    print("   --stratify-by custom_column --balance-classes")
    print("   --stratify-groups group_column --stratify-groups-min-size 5")
    
    print("\n5. Different storage backends:")
    print("   --storage-url postgresql://user:pass@host/db")
    print("   --storage-url mysql://user:pass@host/db")
    
    print("\n6. Remote MLflow tracking:")
    print("   --mlflow-uri http://your-mlflow-server:5000")
    
    print("\nDemo completed! 🎉")


if __name__ == "__main__":
    main()
