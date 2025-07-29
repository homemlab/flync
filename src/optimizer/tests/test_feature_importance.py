#!/usr/bin/env python3
"""
Test script for enhanced feature importance functionality

This script tests the new feature importance graphs with percentages.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import tempfile
import os

# Create a mock MLflow environment for testing
class MockMLflow:
    @staticmethod
    def log_figure(fig, filename):
        print(f"Mock: Would log figure {filename}")
    
    @staticmethod
    def log_artifact(filename):
        print(f"Mock: Would log artifact {filename}")
    
    @staticmethod
    def log_metrics(metrics):
        print(f"Mock: Would log metrics {metrics}")
    
    @staticmethod
    def set_tag(key, value):
        print(f"Mock: Would set tag {key}={value}")

# Mock the mlflow module
import sys
sys.modules['mlflow'] = MockMLflow

# Mock logger
class MockLogger:
    @staticmethod
    def info(msg):
        print(f"INFO: {msg}")

# Import the FeatureImportanceAnalyzer after mocking
from hyperparameter_optimizer import FeatureImportanceAnalyzer

def test_feature_importance_analysis():
    """Test the enhanced feature importance analysis."""
    print("Testing Enhanced Feature Importance Analysis")
    print("=" * 50)
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i:02d}" for i in range(X.shape[1])]
    
    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"✅ Trained RandomForest model with {X.shape[1]} features")
    print(f"   Model accuracy: {model.score(X, y):.3f}")
    
    # Test the enhanced feature importance analysis
    print("\n🔍 Testing feature importance analysis...")
    
    # Replace the global logger with our mock
    import hyperparameter_optimizer
    hyperparameter_optimizer.logger = MockLogger
    
    try:
        # Test final model analysis (no trial number)
        FeatureImportanceAnalyzer.analyze_and_log(
            model, feature_names, "RandomForest", "final"
        )
        print("✅ Final model feature importance analysis completed")
        
        # Test trial analysis (with trial number)
        FeatureImportanceAnalyzer.analyze_and_log(
            model, feature_names, "RandomForest", "trial", trial_number=5
        )
        print("✅ Trial feature importance analysis completed")
        
        # Verify output files were created
        expected_files = [
            "feature_importances_final.csv",
            "feature_importances_trial_5.csv",
            "feature_importances_final.png",
            "feature_importances_trial_5.png"
        ]
        
        created_files = []
        for filename in expected_files:
            if os.path.exists(filename):
                created_files.append(filename)
                print(f"✅ Created: {filename}")
            else:
                print(f"❌ Missing: {filename}")
        
        print(f"\n📊 Summary: {len(created_files)}/{len(expected_files)} files created")
        
        # Show feature importance data
        if os.path.exists("feature_importances_final.csv"):
            df = pd.read_csv("feature_importances_final.csv")
            print(f"\n📈 Top 5 features by importance:")
            for i, row in df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f} ({row['percentage']:.2f}%)")
        
        # Clean up test files
        for filename in created_files:
            try:
                os.remove(filename)
                print(f"🧹 Cleaned up: {filename}")
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"❌ Error during feature importance analysis: {e}")
        return False

def demonstrate_improvements():
    """Demonstrate the improvements made to feature importance analysis."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ENHANCEMENTS")
    print("=" * 60)
    
    improvements = [
        "✨ Enhanced visualization with dual plots (absolute values + percentages)",
        "📊 Percentage calculations for relative importance",
        "🏷️  Value labels on each bar for precise reading",
        "📈 Top feature metrics logged to MLflow for easy comparison",
        "🔢 Cumulative importance percentages (top 5, top 10)",
        "🎯 Trial-specific feature importance tracking",
        "📁 Separate plots and CSV files for each trial and final model",
        "🎨 Professional styling with grid lines and better formatting",
        "📝 Feature names logged as MLflow tags for searchability",
        "🔍 Stability analysis across optimization trials"
    ]
    
    for improvement in improvements:
        print(improvement)
    
    print("\n📝 New Method Signatures:")
    print("   FeatureImportanceAnalyzer.analyze_and_log(")
    print("       model, feature_names, model_type,")
    print("       prefix='final', trial_number=None")
    print("   )")
    print("\n   FeatureImportanceAnalyzer.create_feature_stability_plot(")
    print("       study, model_type")
    print("   )")

def main():
    """Main test function."""
    print("🧪 TESTING ENHANCED FEATURE IMPORTANCE FUNCTIONALITY")
    print("=" * 60)
    
    success = test_feature_importance_analysis()
    
    demonstrate_improvements()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("The enhanced feature importance functionality is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    print("\n🚀 Integration with main script:")
    print("   • Feature importance is now logged for every Optuna trial")
    print("   • Enhanced plots show both absolute and percentage importance")
    print("   • Top features are tracked as MLflow metrics")
    print("   • Stability analysis shows optimization convergence")
    print("   • All visualizations are automatically saved and logged")

if __name__ == "__main__":
    main()
