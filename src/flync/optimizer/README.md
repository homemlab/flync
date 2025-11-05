# FLYNC Model Training & Hyperparameter Optimization

Comprehensive tools for training and optimizing machine learning models for lncRNA classification.

This module provides utilities for:
- **Hyperparameter optimization** using Optuna with multi-objective support
- **Experiment tracking** via MLflow with comprehensive metrics and visualizations
- **Model training** with RandomForest and XGBoost classifiers
- **Feature importance analysis** with stability tracking across trials

---

## Table of Contents

- [Quick Start](#quick-start)
- [Module Overview](#module-overview)
- [Usage Guide](#usage-guide)
  - [Data Preparation](#1-data-preparation)
  - [Hyperparameter Optimization](#2-hyperparameter-optimization)
  - [Viewing Results](#3-viewing-results)
  - [Model Deployment](#4-model-deployment)
- [Command Reference](#command-reference)
- [Optimization Details](#optimization-details)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**Complete training workflow from feature files to optimized model:**

```bash
# 1. Prepare data splits (stratified train/val/test)
python prepare_data.py \
    --positive-file lncrna_features.parquet \
    --negative-file protein_coding_features.parquet \
    --output-dir datasets \
    --target-column y

# 2. Run hyperparameter optimization (RandomForest example)
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type randomforest \
    --optimization-metrics precision f1 \
    --study-name rf_optimization \
    --n-trials 100 \
    --experiment-name "RandomForest_Optimization"

# 3. View results in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in browser

# 4. Extract best model schema for inference
python extract_model_schema.py \
    --model-path best_model.pkl \
    --training-data datasets/train.parquet \
    --output-schema model_schema.json
```

---

## Module Overview

### Files in this Module

**`hyperparameter_optimizer.py`** - Main optimization script
- Optuna-based hyperparameter search
- Multi-objective optimization support
- MLflow experiment tracking
- Comprehensive metric logging
- Feature importance analysis per trial

**`batch_optimization.py`** - Batch training utilities
- Run multiple optimization experiments
- Compare different model configurations
- Automated model selection

**`prepare_data.py`** (referenced, may be in parent directory)
- Stratified train/validation/test splitting
- Balanced class sampling
- Feature preprocessing

### Key Features

✅ **Multiple Model Support**: RandomForest and XGBoost  
✅ **Multi-Objective Optimization**: Optimize for multiple metrics simultaneously  
✅ **Persistent Studies**: Resume interrupted optimizations  
✅ **Comprehensive Tracking**: All metrics, parameters, and artifacts logged to MLflow  
✅ **Feature Importance**: Tracked per trial with stability analysis  
✅ **Visualization Suite**: ROC/PR curves, feature importance plots, optimization history  
✅ **Reproducibility**: Fixed random seeds and versioned datasets  

---

## Usage Guide

### 1. Data Preparation

**From separate positive/negative files:**

```bash
python prepare_data.py \
    --positive-file lncrna_features.parquet \
    --negative-file protein_coding_features.parquet \
    --output-dir datasets \
    --target-column y \
    --train-size 0.7 \
    --val-size 0.15 \
    --test-size 0.15 \
    --random-state 42
```

**Outputs:**
- `datasets/train.parquet` - Training set (70%)
- `datasets/test.parquet` - Validation set (15%)
- `datasets/holdout.parquet` - Final holdout set (15%)

**From single labeled file:**

```bash
python prepare_data.py \
    --dataset labeled_features.parquet \
    --target-column is_lncrna \
    --output-dir datasets \
    --train-size 0.7 --val-size 0.15 --test-size 0.15
```

### 2. Hyperparameter Optimization

**RandomForest optimization:**

```bash
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type randomforest \
    --optimization-metrics precision f1 \
    --optimization-direction maximize \
    --study-name rf_precision_f1 \
    --n-trials 100 \
    --experiment-name "RF_Precision_F1_Optimization" \
    --random-state 42
```

**XGBoost optimization:**

```bash
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type xgboost \
    --optimization-metrics roc_auc pr_auc \
    --optimization-direction maximize \
    --study-name xgb_auc_optimization \
    --n-trials 150 \
    --experiment-name "XGB_AUC_Optimization"
```

**Multi-objective optimization (optimize precision AND recall):**

```bash
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type randomforest \
    --optimization-metrics precision recall \
    --optimization-direction maximize \
    --study-name rf_precision_recall \
    --n-trials 200 \
    --experiment-name "RF_Multi_Objective"
```

### 3. Viewing Results

**Start MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 in your browser.

**In MLflow UI:**
- **Experiments**: View all optimization runs
- **Runs**: Compare metrics across trials
- **Artifacts**: Download models, plots, and feature importance files
- **Metrics**: Sort by accuracy, precision, F1, etc.
- **Parameters**: See hyperparameter values for each trial

**Command-line queries:**

```bash
# List all experiments
mlflow experiments list --tracking-uri sqlite:///mlflow.db

# Get best run for an experiment
mlflow runs list --experiment-name "RF_Precision_F1_Optimization" \
    --sort-by "metrics.final_test_f1 DESC" --max-results 1
```

### 4. Model Deployment

**Extract best model:**

```bash
# Find best run ID from MLflow UI or CLI
export RUN_ID="abc123..."

# Download model artifact
mlflow artifacts download \
    --run-id $RUN_ID \
    --artifact-path "model" \
    --dst-path ./best_model/
```

**Extract model schema for inference:**

```bash
python ../ml/schema_extractor.py \
    --model-path best_model/model.pkl \
    --training-data datasets/train.parquet \
    --output-schema best_model/model_schema.json
```

**Use model for predictions:**

```bash
# Via FLYNC CLI (recommended)
flync run-ml \
    --gtf new_transcripts.gtf \
    --output predictions.csv \
    --ref-genome genome.fa \
    --model best_model/model.pkl

# Or via Python
python -c "
from flync.ml.predictor import predict_from_features
predict_from_features(
    'new_features.parquet',
    'best_model/model.pkl',
    'predictions.csv'
)
"
```

---

## Command Reference

### hyperparameter_optimizer.py

**Required Arguments:**
- `--train-data`: Path to training dataset (parquet)
- `--test-data`: Path to validation dataset (parquet)
- `--holdout-data`: Path to final holdout dataset (parquet)
- `--model-type`: Model type (`randomforest` or `xgboost`)
- `--study-name`: Optuna study name (for resumability)

**Optional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--target-column` | `"y"` | Name of target/label column |
| `--optimization-metrics` | `["precision"]` | Metrics to optimize (see list below) |
| `--optimization-direction` | `"maximize"` | `maximize` or `minimize` |
| `--n-trials` | `100` | Number of optimization trials |
| `--timeout` | `None` | Timeout in seconds |
| `--storage-url` | `"sqlite:///optuna_study.db"` | Optuna database URL |
| `--mlflow-uri` | `"sqlite:///mlflow.db"` | MLflow tracking URI |
| `--experiment-name` | Auto-generated | MLflow experiment name |
| `--random-state` | `42` | Random seed for reproducibility |
| `--project-name` | `None` | Project name for MLflow tags |
| `--dataset-version` | `None` | Dataset version for MLflow tags |

**Available Optimization Metrics:**
- `accuracy` - Overall accuracy
- `precision` - Precision (positive predictive value)
- `recall` - Recall (sensitivity, true positive rate)
- `f1` - F1 score (harmonic mean of precision and recall)
- `roc_auc` - Area under ROC curve
- `pr_auc` - Area under precision-recall curve

**Examples:**

```bash
# Basic usage with defaults
python hyperparameter_optimizer.py \
    --train-data train.parquet \
    --test-data test.parquet \
    --holdout-data holdout.parquet \
    --model-type randomforest \
    --study-name my_study

# Advanced usage with custom settings
python hyperparameter_optimizer.py \
    --train-data train.parquet \
    --test-data test.parquet \
    --holdout-data holdout.parquet \
    --model-type xgboost \
    --optimization-metrics roc_auc pr_auc f1 \
    --study-name xgb_multi_metric \
    --n-trials 200 \
    --timeout 7200 \
    --storage-url postgresql://user:pass@localhost/optuna \
    --mlflow-uri http://mlflow-server:5000 \
    --experiment-name "XGB_Production_v2" \
    --project-name "lncRNA_Discovery" \
    --dataset-version "v2.1"
```

---

## Optimization Details

### Hyperparameter Search Spaces

**RandomForest:**
```python
{
    'n_estimators': (50, 1000),          # Log scale
    'max_depth': (3, 50),                # Integer
    'min_samples_split': (0.01, 1.0),   # Float
    'min_samples_leaf': (0.01, 0.5),    # Float
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'criterion': ['gini', 'entropy']
}
```

**XGBoost:**
```python
{
    'n_estimators': (50, 1500),          # Log scale
    'learning_rate': (1e-4, 0.3),       # Log scale
    'max_depth': (3, 12),               # Integer
    'min_child_weight': (1, 20),        # Integer
    'gamma': (0.0, 1.0),                # Float
    'subsample': (0.5, 1.0),            # Float
    'colsample_bytree': (0.5, 1.0),     # Float
    'reg_alpha': (1e-8, 10.0),          # Log scale (L1)
    'reg_lambda': (1e-8, 10.0),         # Log scale (L2)
    'scale_pos_weight': (1.0, 20.0)     # Class imbalance
}
```

### Optimization Process

For each trial:

1. **Sample Hyperparameters**: Optuna samples from search space
2. **Train Model**: Fit model on training data
3. **Validation**: Evaluate on validation (test) data
4. **Metric Calculation**: Compute all specified metrics
5. **Feature Importance**: Extract and log feature importances
6. **Logging**: Save metrics, params, and artifacts to MLflow
7. **Objective Score**: Return metric(s) for optimization

**Multi-Objective Optimization:**
- When multiple metrics specified, Optuna uses non-dominated sorting
- Returns Pareto-optimal solutions
- View tradeoffs in Optuna visualizations

### Resume Capability

Optuna studies are persistent - you can resume interrupted optimizations:

```bash
# Initial run (completes 50 trials)
python hyperparameter_optimizer.py \
    --study-name my_study \
    --n-trials 50 \
    [other args...]

# Resume with more trials (same study name!)
python hyperparameter_optimizer.py \
    --study-name my_study \
    --n-trials 100 \
    [same other args...]
```

The study will continue from trial 51.

---

## Output Files

### MLflow Artifacts

Logged for **every trial**:
- `feature_importances_trial_N.png` - Feature importance plot with percentages
- `feature_importances_trial_N.csv` - Feature importance data
- `roc_curve_trial_N.png` - ROC curve on validation data
- `pr_curve_trial_N.png` - Precision-recall curve

Logged for **final holdout evaluation** (best model):
- `feature_importances_final.png` - Final feature importance plot
- `feature_importances_final.csv` - Final feature importance data
- `roc_curve_final.png` - ROC curve on holdout data
- `pr_curve_final.png` - PR curve on holdout data
- `model/` - Serialized trained model

### Optuna Visualizations

Generated during optimization:
- `optimization_history_<model>.png` - Objective value over trials
- `param_importances_<model>.png` - Parameter importance
- `optimization_stability_<model>.png` - Feature importance stability

### Databases

- **`optuna_study.db`**: Optuna study database (SQLite by default)
  - Contains all trials, parameters, and intermediate values
  - Persistent across runs
  - Can be PostgreSQL/MySQL for production

- **`mlflow.db`**: MLflow tracking database (SQLite by default)
  - Contains experiments, runs, metrics, parameters
  - Artifacts stored in `mlruns/` directory
  - Can be remote server for team collaboration

### Log Files

- **`hyperparameter_optimization.log`**: Detailed execution log
  - All INFO, WARNING, ERROR messages
  - Timestamps for debugging
  - Trial-by-trial progress

---

## Advanced Usage

### Custom Storage Backends

**PostgreSQL (recommended for production):**

```bash
# Setup database
createdb optuna_db
createdb mlflow_db

# Run optimization
python hyperparameter_optimizer.py \
    --storage-url postgresql://user:password@localhost/optuna_db \
    --mlflow-uri postgresql://user:password@localhost/mlflow_db \
    [other args...]
```

**MySQL:**

```bash
python hyperparameter_optimizer.py \
    --storage-url mysql://user:password@localhost/optuna_db \
    --mlflow-uri mysql://user:password@localhost/mlflow_db \
    [other args...]
```

**Remote MLflow server:**

```bash
# Start MLflow server on remote machine
mlflow server --host 0.0.0.0 --port 5000

# Point optimizer to remote server
python hyperparameter_optimizer.py \
    --mlflow-uri http://remote-server:5000 \
    [other args...]
```

### Distributed Optimization

Run multiple workers in parallel (same study):

```bash
# Terminal 1
python hyperparameter_optimizer.py \
    --study-name shared_study \
    --n-trials 50 \
    --storage-url postgresql://localhost/optuna \
    [other args...]

# Terminal 2 (same study name and storage!)
python hyperparameter_optimizer.py \
    --study-name shared_study \
    --n-trials 50 \
    --storage-url postgresql://localhost/optuna \
    [other args...]
```

Optuna coordinates trials across workers automatically.

### Batch Optimization

Compare multiple configurations:

```bash
# Create configuration file
cat > optimization_configs.yaml << EOF
experiments:
  - name: "RF_Precision"
    model_type: randomforest
    metrics: [precision]
    trials: 100
  
  - name: "RF_F1"
    model_type: randomforest
    metrics: [f1]
    trials: 100
  
  - name: "XGB_AUC"
    model_type: xgboost
    metrics: [roc_auc]
    trials: 150
EOF

# Run batch optimization (if batch_optimization.py supports config files)
python batch_optimization.py --config optimization_configs.yaml
```

### Feature Importance Analysis

**Extract top features across all trials:**

```python
import mlflow
import pandas as pd

# Connect to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.get_experiment_by_name("My_Experiment")

# Get all runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Extract top feature metrics
top_features = runs[[
    'metrics.trial_top_1_feature_importance',
    'metrics.trial_top_5_cumulative_percentage',
    'tags.trial_top_1_feature_name'
]].sort_values('metrics.trial_top_1_feature_importance', ascending=False)

print(top_features.head(10))
```

**Stability analysis:**

```python
import matplotlib.pyplot as plt

# Plot feature importance convergence
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(runs['metrics.trial_number'], 
        runs['metrics.trial_top_1_feature_importance'])
ax.set_xlabel('Trial')
ax.set_ylabel('Top Feature Importance')
ax.set_title('Feature Importance Stability')
plt.savefig('feature_stability.png')
```

---

## Troubleshooting

### Common Issues

**Problem**: `sqlite3.OperationalError: database is locked`
```bash
# Solution 1: Use PostgreSQL instead of SQLite for concurrent access
--storage-url postgresql://localhost/optuna_db

# Solution 2: Ensure no other processes are using the database
lsof optuna_study.db  # Check what's using it
```

**Problem**: Memory error during training
```bash
# Solution 1: Reduce number of estimators in search space
# Edit hyperparameter_optimizer.py and reduce n_estimators max

# Solution 2: Reduce dataset size
python prepare_data.py --train-size 0.5  # Use 50% instead of 70%

# Solution 3: Use fewer features
# Filter features before optimization
```

**Problem**: MLflow port already in use
```bash
# Solution: Use different port
mlflow ui --port 5001

# Or kill process using port 5000
lsof -ti:5000 | xargs kill -9
```

**Problem**: Optimization very slow
```bash
# Solution 1: Reduce number of trials
--n-trials 50  # Instead of 100

# Solution 2: Use timeout instead
--timeout 3600  # 1 hour

# Solution 3: Simplify search space
# Edit hyperparameter_optimizer.py to reduce parameter ranges

# Solution 4: Use smaller validation set
# Create smaller test.parquet
```

**Problem**: No improvement after many trials
```bash
# Solution 1: Check data quality
# Ensure features are informative and labels are correct

# Solution 2: Try different model type
--model-type xgboost  # If RandomForest not working

# Solution 3: Add more features
# Extract additional genomic features

# Solution 4: Adjust search space
# Expand ranges for hyperparameters that seem important
```

**Problem**: Feature importance plots missing
```bash
# Solution: Ensure matplotlib backend is set
export MPLBACKEND=Agg
python hyperparameter_optimizer.py [args...]

# Or install required dependencies
conda install matplotlib seaborn
```

### Debug Mode

Enable verbose logging:

```bash
# Set log level in script or via environment
export FLYNC_LOG_LEVEL=DEBUG
python hyperparameter_optimizer.py [args...]

# Or edit hyperparameter_optimizer.py:
# logging.basicConfig(level=logging.DEBUG)
```

Check logs:
```bash
tail -f hyperparameter_optimization.log
```

---

## Performance Tips

1. **Start Small**: Begin with 50-100 trials to validate setup
2. **Use Persistent Storage**: Always use database (not in-memory) for Optuna
3. **Monitor Resources**: Watch CPU and memory during optimization
4. **Stratified Splits**: Ensure balanced classes in train/val/test
5. **Feature Selection**: Remove low-importance features before optimization
6. **Early Stopping**: Use timeout for time-bounded optimization
7. **Distributed Workers**: Run multiple workers for faster completion
8. **Cache Features**: Pre-extract features once, reuse for multiple experiments

---

## Integration with FLYNC

This module integrates with the main FLYNC pipeline:

```bash
# 1. Extract features using FLYNC
flync run-ml --gtf transcripts.gtf --extract-only --output features.parquet

# 2. Train custom model (this module)
python prepare_data.py --positive-file lncrna_features.parquet ...
python hyperparameter_optimizer.py --train-data train.parquet ...

# 3. Deploy model back to FLYNC
flync run-ml --gtf new_transcripts.gtf --model best_model.pkl --output predictions.csv
```

---

**Module**: `src/flync/optimizer/`  
**Last Updated**: November 2025  
**Version**: 1.0.0

For more information, see:
- Main documentation: [../../README.md](../../README.md)
- Feature extraction: [../features/README.md](../features/README.md)
- ML prediction: [../ml/README.md](../ml/README.md) (if exists)
