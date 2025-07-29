#!/usr/bin/env python3
"""
Simple run script for hyperparameter optimization

This script provides example commands for running the optimization pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Run a shell command and handle errors."""
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main function to run the optimization pipeline."""
    
    # Check if data preparation is needed
    datasets_dir = Path("./datasets")
    if not datasets_dir.exists() or not all(
        (datasets_dir / f).exists() for f in ["train.parquet", "test.parquet", "holdout.parquet"]
    ):
        print("Preparing datasets...")
        prep_command = [
            "python", "../prepare_data.py",
            "--positive-file", "../../ncr_dim_redux.parquet",
            "--negative-file", "../../pcg_dim_redux.parquet", 
            "--output-dir", "datasets",
            "--target-column", "y"
        ]
        
        if not run_command(prep_command):
            print("Data preparation failed. Exiting.")
            sys.exit(1)
    else:
        print("✓ Datasets already prepared")
    
    # Run hyperparameter optimization
    print("\nRunning hyperparameter optimization...")
    
    # Example with RandomForest optimizing for precision
    rf_command = [
        "python", "../hyperparameter_optimizer.py",
        "--train-data", "datasets/train.parquet",
        "--test-data", "datasets/test.parquet", 
        "--holdout-data", "datasets/holdout.parquet",
        "--target-column", "y",
        "--model-type", "randomforest",
        "--optimization-metrics", "precision", "f1",
        "--optimization-direction", "maximize",
        "--study-name", "rf_precision_f1_study",
        "--n-trials", "50",  # Reduced for demo
        "--experiment-name", "RandomForest_Optimization_Demo"
    ]
    
    if not run_command(rf_command):
        print("RandomForest optimization failed.")
        return False
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print("To view results:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("\nTo run with XGBoost:")
    print("  python ../hyperparameter_optimizer.py \\")
    print("    --train-data datasets/train.parquet \\")
    print("    --test-data datasets/test.parquet \\") 
    print("    --holdout-data datasets/holdout.parquet \\")
    print("    --model-type xgboost \\")
    print("    --optimization-metrics precision recall \\")
    print("    --study-name xgb_precision_recall_study")
    
    return True

if __name__ == "__main__":
    main()
