"""
Hyperparameter Optimization Package

A comprehensive suite of tools for machine learning hyperparameter optimization
using Optuna and MLflow with advanced features like stratified data splitting,
feature importance analysis, and experiment tracking.

Key Components:
- hyperparameter_optimizer.py: Main optimization engine
- prepare_data.py: Advanced data preparation with stratification
- demo.py: Interactive demonstration script
- feature_importance_demo.py: Feature importance showcase
"""

__version__ = "1.0.0"
__author__ = "ML Optimization Team"

# Package metadata
PACKAGE_NAME = "hyperparameter-optimizer"
DESCRIPTION = "Advanced hyperparameter optimization toolkit with MLflow and Optuna"

# Version history
VERSION_HISTORY = {
    "1.0.0": "Initial release with enhanced feature importance and stratified splitting",
    "0.9.0": "Beta release with basic optimization functionality",
}

# Supported models
SUPPORTED_MODELS = ["randomforest", "xgboost"]

# Supported metrics
SUPPORTED_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
