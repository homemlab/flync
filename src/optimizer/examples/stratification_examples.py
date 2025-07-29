#!/usr/bin/env python3
"""
Stratified Data Preparation Examples

This script demonstrates the enhanced stratification features in the prepare_data.py script.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def run_example(name, command, description):
    """Run a data preparation example."""
    print(f"\n📊 Example: {name}")
    print(f"Description: {description}")
    print(f"Command: {' '.join(command)}")
    
    user_input = input(f"\nRun {name} example? (y/n): ")
    if user_input.lower() == 'y':
        try:
            subprocess.run(command, check=True)
            print(f"✅ {name} example completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {name} example failed: {e}")
            return False
    else:
        print(f"⏭️  Skipping {name} example")
        return None


def main():
    """Main demonstration function."""
    
    print_section("STRATIFIED DATA PREPARATION EXAMPLES")
    
    # Check for data files
    pos_file = "../../ncr_dim_redux.parquet"
    neg_file = "../../pcg_dim_redux.parquet"
    
    if not (Path(pos_file).exists() and Path(neg_file).exists()):
        print(f"⚠️  Data files not found: {pos_file}, {neg_file}")
        print("Please ensure the original data files are in the parent directory.")
        return
    
    print("✅ Found original data files")
    
    examples = [
        {
            "name": "Basic Stratified Split",
            "description": "Standard stratification by target column",
            "command": [
                "python", "../prepare_data.py",
                "--positive-file", pos_file,
                "--negative-file", neg_file,
                "--output-dir", "datasets_basic",
                "--target-column", "y"
            ]
        },
        {
            "name": "Balanced Classes",
            "description": "Balance classes by undersampling majority class",
            "command": [
                "python", "../prepare_data.py",
                "--positive-file", pos_file,
                "--negative-file", neg_file,
                "--output-dir", "datasets_balanced",
                "--target-column", "y",
                "--balance-classes",
                "--max-imbalance-ratio", "3.0"
            ]
        },
        {
            "name": "Custom Stratification Column",
            "description": "Stratify by a custom column (if available)",
            "command": [
                "python", "../prepare_data.py",
                "--positive-file", pos_file,
                "--negative-file", neg_file,
                "--output-dir", "datasets_custom_strat",
                "--target-column", "y",
                "--stratify-by", "y"  # Can be changed to any available column
            ]
        },
        {
            "name": "Group-Aware Stratification",
            "description": "Stratified splits that respect group boundaries",
            "command": [
                "python", "../prepare_data.py",
                "--positive-file", pos_file,
                "--negative-file", neg_file,
                "--output-dir", "datasets_group_strat",
                "--target-column", "y",
                "--stratify-groups", "gene_name",  # Assuming this column exists
                "--stratify-groups-min-size", "5"
            ]
        },
        {
            "name": "Combined Advanced Options",
            "description": "Custom stratification + class balancing + group awareness",
            "command": [
                "python", "../prepare_data.py",
                "--positive-file", pos_file,
                "--negative-file", neg_file,
                "--output-dir", "datasets_advanced",
                "--target-column", "y",
                "--balance-classes",
                "--max-imbalance-ratio", "5.0",
                "--test-size", "0.25",
                "--holdout-size", "0.20",
                "--random-state", "123"
            ]
        }
    ]
    
    print_section("STRATIFICATION EXAMPLES")
    
    for example in examples:
        success = run_example(example["name"], example["command"], example["description"])
        if success is False:
            print("⚠️  Some examples may fail if required columns don't exist in your data")
    
    print_section("PARAMETER EXPLANATIONS")
    
    explanations = [
        ("--stratify-by COLUMN", "Use COLUMN for stratification instead of target"),
        ("--stratify-groups COLUMN", "Ensure samples with same COLUMN value stay together"),
        ("--stratify-groups-min-size N", "Minimum group size for group stratification"),
        ("--balance-classes", "Undersample majority class to balance dataset"),
        ("--max-imbalance-ratio X", "Maximum allowed ratio between majority/minority"),
        ("--test-size 0.X", "Fraction of data for test set"),
        ("--holdout-size 0.X", "Fraction of data for holdout set"),
        ("--random-state N", "Random seed for reproducible splits")
    ]
    
    for param, explanation in explanations:
        print(f"  {param:30} - {explanation}")
    
    print_section("USE CASES")
    
    use_cases = [
        {
            "scenario": "📊 Time Series Data",
            "recommendation": "Use --stratify-groups with time periods to prevent data leakage"
        },
        {
            "scenario": "👥 Patient/User Data", 
            "recommendation": "Use --stratify-groups with patient/user IDs to avoid overfitting"
        },
        {
            "scenario": "🏢 Hierarchical Data",
            "recommendation": "Use --stratify-groups with organization/location IDs"
        },
        {
            "scenario": "⚖️ Imbalanced Classes",
            "recommendation": "Use --balance-classes with appropriate --max-imbalance-ratio"
        },
        {
            "scenario": "🎯 Custom Categories",
            "recommendation": "Use --stratify-by with categorical features for better representation"
        }
    ]
    
    for use_case in use_cases:
        print(f"{use_case['scenario']}: {use_case['recommendation']}")
    
    print_section("INTEGRATION WITH OPTIMIZATION")
    
    print("After preparing your data with stratification, use it in optimization:")
    print()
    print("python hyperparameter_optimizer.py \\")
    print("  --train-data datasets_advanced/train.parquet \\")
    print("  --test-data datasets_advanced/test.parquet \\")
    print("  --holdout-data datasets_advanced/holdout.parquet \\")
    print("  --model-type randomforest \\")
    print("  --optimization-metrics precision f1 \\")
    print("  --study-name stratified_data_study")
    
    print("\n🎯 Benefits of stratified splitting:")
    benefits = [
        "✅ Maintains class distribution across splits",
        "✅ Prevents data leakage in grouped data",
        "✅ Handles class imbalance appropriately", 
        "✅ Ensures representative test/validation sets",
        "✅ Improves model evaluation reliability"
    ]
    
    for benefit in benefits:
        print(benefit)


if __name__ == "__main__":
    main()
