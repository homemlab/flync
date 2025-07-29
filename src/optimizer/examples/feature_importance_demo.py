#!/usr/bin/env python3
"""
Feature Importance Demo

This script demonstrates the enhanced feature importance functionality
without requiring the full hyperparameter optimization pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_demo_with_existing_data():
    """Run a quick demo using existing prepared data."""
    
    # Check if prepared datasets exist
    datasets_dir = Path("datasets")
    required_files = ["train.parquet", "test.parquet", "holdout.parquet"]
    
    if not datasets_dir.exists():
        logger.error("datasets/ directory not found. Please run data preparation first.")
        return False
    
    missing_files = []
    for file in required_files:
        if not (datasets_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        logger.info("Please run: python prepare_data.py --positive-file ncr_dim_redux.parquet --negative-file pcg_dim_redux.parquet --output-dir datasets")
        return False
    
    logger.info("✅ Found prepared datasets")
    
    # Run a quick optimization with enhanced feature importance
    cmd = [
        "python", "hyperparameter_optimizer.py",
        "--train-data", "datasets/train.parquet",
        "--test-data", "datasets/test.parquet",
        "--holdout-data", "datasets/holdout.parquet",
        "--target-column", "y",
        "--model-type", "randomforest",
        "--optimization-metrics", "precision",
        "--study-name", "feature_importance_demo",
        "--n-trials", "5",  # Quick demo with just 5 trials
        "--experiment-name", "Feature_Importance_Demo"
    ]
    
    logger.info("Running feature importance demo...")
    logger.info("Command: " + " ".join(cmd))
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ Demo completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Demo failed: {e}")
        if e.stdout:
            logger.info(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def show_expected_outputs():
    """Show what outputs the user should expect to see."""
    
    print("\n" + "="*60)
    print("EXPECTED FEATURE IMPORTANCE OUTPUTS")
    print("="*60)
    
    outputs = [
        {
            "category": "📊 Enhanced Plots (per trial)",
            "files": [
                "feature_importances_trial_0.png - Dual plot (absolute + percentage)",
                "feature_importances_trial_1.png",
                "... (one per trial)",
                "feature_importances_final.png - Final model analysis"
            ]
        },
        {
            "category": "📋 Data Files (per trial)", 
            "files": [
                "feature_importances_trial_0.csv - With percentage column",
                "feature_importances_trial_1.csv",
                "... (one per trial)",
                "feature_importances_final.csv - Final model data"
            ]
        },
        {
            "category": "📈 MLflow Metrics (per trial)",
            "files": [
                "trial_top_1_feature_importance - Value of most important feature",
                "trial_top_1_feature_percentage - Percentage of most important feature", 
                "trial_top_5_cumulative_percentage - Coverage by top 5 features",
                "final_top_10_cumulative_percentage - Final model coverage"
            ]
        },
        {
            "category": "🏷️ MLflow Tags (per trial)",
            "files": [
                "trial_top_1_feature_name - Name of most important feature",
                "trial_top_2_feature_name - Name of 2nd most important feature",
                "... (top 5 features per trial)"
            ]
        },
        {
            "category": "📉 Stability Analysis",
            "files": [
                "optimization_stability_randomforest.png - Convergence plot",
                "Shows how optimization score evolves across trials"
            ]
        }
    ]
    
    for output in outputs:
        print(f"\n{output['category']}:")
        for file in output['files']:
            print(f"  • {file}")

def show_mlflow_instructions():
    """Show how to view the results in MLflow."""
    
    print("\n" + "="*60)  
    print("VIEWING RESULTS IN MLFLOW")
    print("="*60)
    
    instructions = [
        "1. Start MLflow UI:",
        "   mlflow ui --backend-store-uri sqlite:///mlflow.db",
        "",
        "2. Open browser to: http://localhost:5000",
        "",
        "3. Navigate to 'Feature_Importance_Demo' experiment",
        "",
        "4. Click on the main run to see:",
        "   • Final model feature importance plot",
        "   • Stability analysis plot", 
        "   • Feature importance metrics",
        "",
        "5. Click on individual trial runs to see:",
        "   • Trial-specific feature importance plots",
        "   • Top feature metrics for that trial",
        "   • Feature name tags",
        "",
        "6. Compare across trials:",
        "   • Use the 'Compare' feature to see metric evolution",
        "   • Look for consistency in top features",
        "   • Track cumulative percentage trends"
    ]
    
    for instruction in instructions:
        print(instruction)

def main():
    """Main demo function."""
    
    parser = argparse.ArgumentParser(description="Feature Importance Enhancement Demo")
    parser.add_argument("--prepare-data", action="store_true", 
                       help="Prepare data first (requires original parquet files)")
    args = parser.parse_args()
    
    print("🎯 FEATURE IMPORTANCE ENHANCEMENT DEMO")
    print("="*60)
    
    if args.prepare_data:
        logger.info("Preparing datasets first...")
        prep_cmd = [
            "python", "prepare_data.py",
            "--positive-file", "../ncr_dim_redux.parquet",
            "--negative-file", "../pcg_dim_redux.parquet",
            "--output-dir", "datasets"
        ]
        
        import subprocess
        try:
            subprocess.run(prep_cmd, check=True)
            logger.info("✅ Data preparation completed")
        except subprocess.CalledProcessError:
            logger.error("❌ Data preparation failed")
            logger.info("Make sure ../ncr_dim_redux.parquet and ../pcg_dim_redux.parquet exist")
            sys.exit(1)
    
    # Run the demo
    success = run_demo_with_existing_data()
    
    show_expected_outputs()
    show_mlflow_instructions()
    
    print("\n" + "="*60)
    if success:
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("\nKey Enhancements Demonstrated:")
        enhancements = [
            "• Enhanced plots with absolute values AND percentages",
            "• Value labels on bars for precise reading",
            "• Trial-specific feature importance tracking",
            "• MLflow metrics for top features",
            "• Feature names as searchable tags",
            "• Cumulative importance percentages",
            "• Stability analysis across trials",
            "• Professional visualization styling"
        ]
        for enhancement in enhancements:
            print(enhancement)
        
        print(f"\n🚀 Next steps:")
        print("   • Start MLflow UI to explore results")
        print("   • Compare feature importance across trials")
        print("   • Look for consistent top features")
        print("   • Use insights for feature selection")
        
    else:
        print("❌ DEMO FAILED!")
        print("Please check the error messages above.")
        
    print("\n💡 For production runs, use more trials (--n-trials 100+)")

if __name__ == "__main__":
    main()
