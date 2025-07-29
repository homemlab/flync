#!/usr/bin/env python3
"""
Usage Examples - Before vs After

This script shows the difference between the original script and the improved version.
"""

def show_original_usage():
    """Show how the original script was used."""
    print("="*60)
    print("ORIGINAL SCRIPT USAGE")
    print("="*60)
    
    print("❌ Problems with the original script:")
    problems = [
        "• Hard-coded data loading and preprocessing",
        "• No CLI interface - had to edit source code",
        "• Fixed optimization metric (precision only)",
        "• Mixed data preparation and optimization logic",
        "• No easy way to resume studies",
        "• Limited configurability",
        "• No proper error handling",
        "• Difficult to use with different datasets"
    ]
    
    for problem in problems:
        print(problem)
    
    print("\n📝 To use the original script:")
    print("1. Edit main.py to change:")
    print("   - Data file paths")
    print("   - Model type")
    print("   - Number of trials")
    print("   - Study name")
    print("   - Other parameters")
    print("2. Run: python main.py")
    print("3. Hope nothing breaks!")


def show_improved_usage():
    """Show how the improved script is used."""
    print("\n" + "="*60)
    print("IMPROVED SCRIPT USAGE")
    print("="*60)
    
    print("✅ Improvements in the new version:")
    improvements = [
        "• Full CLI interface with argparse",
        "• Modular, object-oriented design",
        "• Flexible data input (any parquet files)",
        "• Multiple optimization metrics support",
        "• Easy study resumption",
        "• Comprehensive error handling and logging",
        "• Professional code structure with type hints",
        "• Configurable hyperparameter spaces",
        "• Better MLflow integration",
        "• Feature importance analysis",
        "• Proper train/test/holdout workflow"
    ]
    
    for improvement in improvements:
        print(improvement)
    
    print("\n📝 To use the improved script:")
    print("1. Prepare your data (one-time):")
    print("   python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet --output-dir datasets")
    
    print("\n2. Run optimization with any configuration:")
    print("   python hyperparameter_optimizer.py \\")
    print("     --train-data datasets/train.parquet \\")
    print("     --test-data datasets/test.parquet \\")
    print("     --holdout-data datasets/holdout.parquet \\")
    print("     --model-type randomforest \\")
    print("     --optimization-metrics precision f1 recall \\")
    print("     --study-name my_study \\")
    print("     --n-trials 100")
    
    print("\n3. View results:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")


def show_examples():
    """Show specific usage examples."""
    print("\n" + "="*60)
    print("SPECIFIC USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "name": "Basic RandomForest optimization",
            "command": """python hyperparameter_optimizer.py \\
  --train-data datasets/train.parquet \\
  --test-data datasets/test.parquet \\
  --holdout-data datasets/holdout.parquet \\
  --model-type randomforest \\
  --study-name basic_rf_study"""
        },
        {
            "name": "Multi-metric XGBoost optimization",
            "command": """python hyperparameter_optimizer.py \\
  --train-data datasets/train.parquet \\
  --test-data datasets/test.parquet \\
  --holdout-data datasets/holdout.parquet \\
  --model-type xgboost \\
  --optimization-metrics precision recall f1 \\
  --study-name multi_metric_xgb_study \\
  --n-trials 200"""
        },
        {
            "name": "Production run with timeout",
            "command": """python hyperparameter_optimizer.py \\
  --train-data datasets/train.parquet \\
  --test-data datasets/test.parquet \\
  --holdout-data datasets/holdout.parquet \\
  --model-type randomforest \\
  --optimization-metrics precision \\
  --study-name production_rf_study \\
  --n-trials 500 \\
  --timeout 3600 \\
  --project-name "Production Model" \\
  --dataset-version "v2.1" \\
  --experiment-name "Production_RF_Optimization_v2.1" """
        },
        {
            "name": "Resume existing study",
            "command": """# First run
python hyperparameter_optimizer.py \\
  --study-name my_study \\
  --n-trials 100 \\
  [other args...]

# Resume with more trials
python hyperparameter_optimizer.py \\
  --study-name my_study \\  # Same name!
  --n-trials 200 \\         # Additional trials
  [same other args...]"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(example['command'])


def show_file_structure():
    """Show the improved file structure."""
    print("\n" + "="*60)
    print("FILE STRUCTURE COMPARISON")
    print("="*60)
    
    print("📁 Original structure:")
    original = [
        "main.py                     # Everything in one file",
        "ncr_dim_redux.parquet       # Data files",
        "pcg_dim_redux.parquet",
        "requirements.txt (basic)"
    ]
    for file in original:
        print(f"  {file}")
    
    print("\n📁 Improved structure:")
    improved = [
        "hyperparameter_optimizer.py # Main optimization script (modular)",
        "prepare_data.py             # Data preparation utility",
        "run_optimization.py         # Simple run script",
        "demo.py                     # Interactive demo",
        "requirements.txt            # Comprehensive dependencies",
        "config_example.py           # Configuration example",
        "Makefile                    # Common tasks automation",
        "README_new.md               # Comprehensive documentation",
        "datasets/                   # Prepared datasets directory",
        "  ├── train.parquet",
        "  ├── test.parquet", 
        "  └── holdout.parquet"
    ]
    for file in improved:
        print(f"  {file}")


def main():
    """Main function to show all comparisons."""
    print("HYPERPARAMETER OPTIMIZATION TOOL")
    print("Before vs After Comparison")
    
    show_original_usage()
    show_improved_usage()
    show_examples()
    show_file_structure()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The improved version transforms a monolithic script into a")
    print("professional, configurable, and maintainable CLI tool that")
    print("follows best practices for ML experimentation and can be")
    print("easily integrated into production workflows.")
    
    print("\n🚀 Ready to use? Run:")
    print("  python demo.py")
    print("  or")
    print("  make demo")


if __name__ == "__main__":
    main()
