# Hyperparameter Optimization CLI Tool

A comprehensive command-line tool for hyperparameter optimization of machine learning models using Optuna and MLflow. This tool supports RandomForest and XGBoost classifiers with multiple optimization metrics and provides full experiment tracking.

## Features

- **Multiple Model Support**: RandomForest and XGBoost classifiers
- **Flexible Optimization**: Optimize for multiple metrics (precision, recall, f1, accuracy, roc_auc, pr_auc)
- **Experiment Tracking**: Full MLflow integration with visualization
- **Persistent Storage**: Optuna studies stored in database for resumability
- **CLI-Based**: Easy-to-use command-line interface
- **Comprehensive Logging**: Detailed logging and error handling
- **Enhanced Feature Importance**: Advanced feature importance analysis with percentages and trial tracking
- **Rich Visualizations**: ROC curves, PR curves, feature importance plots, and Optuna optimization plots
- **Stability Analysis**: Track feature importance stability across optimization trials

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

If you have separate positive and negative sample files:

```bash
python prepare_data.py \
    --positive-file ncr_dim_redux.parquet \
    --negative-file pcg_dim_redux.parquet \
    --output-dir datasets \
    --target-column y
```

This will create `train.parquet`, `test.parquet`, and `holdout.parquet` in the `datasets/` directory.

### 2. Run Hyperparameter Optimization

#### RandomForest Example:
```bash
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type randomforest \
    --optimization-metrics precision f1 \
    --optimization-direction maximize \
    --study-name rf_precision_f1_study \
    --n-trials 100 \
    --experiment-name "RandomForest_Precision_F1_Optimization"
```

#### XGBoost Example:
```bash
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type xgboost \
    --optimization-metrics precision recall \
    --optimization-direction maximize \
    --study-name xgb_precision_recall_study \
    --n-trials 100 \
    --experiment-name "XGBoost_Precision_Recall_Optimization"
```

### 3. View Results

Start MLflow UI to view results:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 in your browser.

## Usage

### Command Line Arguments

#### Required Arguments:
- `--train-data`: Path to training dataset (parquet format)
- `--test-data`: Path to test dataset (parquet format) 
- `--holdout-data`: Path to holdout dataset (parquet format)
- `--model-type`: Type of model (`randomforest` or `xgboost`)
- `--study-name`: Name for the Optuna study

#### Optional Arguments:
- `--target-column`: Name of target column (default: "y")
- `--optimization-metrics`: Metrics to optimize (default: ["precision"])
  - Available: accuracy, precision, recall, f1, roc_auc, pr_auc
- `--optimization-direction`: Direction of optimization (default: "maximize")
- `--storage-url`: Database URL for Optuna storage (default: "sqlite:///optuna_study.db")
- `--n-trials`: Number of optimization trials (default: 100)
- `--timeout`: Timeout in seconds
- `--mlflow-uri`: MLflow tracking URI (default: "sqlite:///mlflow.db")
- `--experiment-name`: MLflow experiment name (auto-generated if not provided)
- `--random-state`: Random state for reproducibility (default: 42)
- `--project-name`: Project name for MLflow tags
- `--dataset-version`: Dataset version for MLflow tags

### Example with All Options:
```bash
python hyperparameter_optimizer.py \
    --train-data datasets/train.parquet \
    --test-data datasets/test.parquet \
    --holdout-data datasets/holdout.parquet \
    --target-column y \
    --model-type randomforest \
    --optimization-metrics precision recall f1 \
    --optimization-direction maximize \
    --study-name comprehensive_rf_study \
    --storage-url sqlite:///studies.db \
    --n-trials 200 \
    --timeout 7200 \
    --mlflow-uri sqlite:///experiments.db \
    --experiment-name "Comprehensive_RF_Study" \
    --random-state 42 \
    --project-name "Binary Classification Project" \
    --dataset-version "v2.1"
```

## File Structure

```
├── hyperparameter_optimizer.py    # Main optimization script
├── prepare_data.py                # Data preparation utility
├── requirements.txt               # Python dependencies
├── Makefile                       # Build automation
├── README.md                      # This file
├── examples/                      # Example and demo scripts
│   ├── demo.py                    # Interactive demo
│   ├── feature_importance_demo.py # Feature importance demo
│   ├── stratification_examples.py # Stratification examples
│   ├── run_optimization.py       # Simple run script
│   ├── comparison.py              # Model comparison utilities
│   └── config_example.py          # Configuration example
└── tests/                         # Test scripts
    ├── test_feature_importance.py # Feature importance tests
    └── test_stratification.py     # Stratification tests
```

## Output Files

The tool creates several output files:

- `optuna_study.db`: Optuna study database (persistent)
- `mlflow.db`: MLflow tracking database
- `mlruns/`: MLflow artifacts directory
- `hyperparameter_optimization.log`: Detailed log file
- `feature_importances.png`: Feature importance plot
- `feature_importances.csv`: Feature importance data

## Testing and Examples

The optimizer includes comprehensive testing and example scripts:

```bash
# Run stratification feature tests
make test-stratification

# Run interactive stratification examples
make stratification-demo

# Test feature importance functionality
make test-features

# Run interactive demo
make demo

# Run feature importance demo
make feature-demo
```

## Advanced Usage

### Resuming Studies

Optuna studies are persistent and can be resumed:

```bash
# Run initial optimization
python hyperparameter_optimizer.py \
    --study-name my_study \
    --n-trials 50 \
    [other args...]

# Resume with more trials
python hyperparameter_optimizer.py \
    --study-name my_study \  # Same study name
    --n-trials 100 \         # Additional trials
    [same other args...]
```

### Custom Storage Backends

#### PostgreSQL:
```bash
--storage-url "postgresql://user:password@localhost/optuna_db"
```

#### MySQL:
```bash
--storage-url "mysql://user:password@localhost/optuna_db"
```

### Remote MLflow Tracking:
```bash
--mlflow-uri "http://your-mlflow-server:5000"
```

## Hyperparameter Search Spaces

### RandomForest:
- `n_estimators`: 50-1000 (log scale)
- `max_depth`: 3-50
- `min_samples_split`: 0.01-1.0
- `min_samples_leaf`: 0.01-0.5
- `max_features`: ["sqrt", "log2", None]
- `bootstrap`: [True, False]
- `class_weight`: [None, "balanced", "balanced_subsample"]
- `criterion`: ["gini", "entropy"]

### XGBoost:
- `n_estimators`: 50-1500 (log scale)
- `learning_rate`: 1e-4 to 0.3 (log scale)
- `max_depth`: 3-12
- `min_child_weight`: 1-20
- `gamma`: 0.0-1.0
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0
- `reg_alpha`: 1e-8 to 10.0 (log scale)
- `reg_lambda`: 1e-8 to 10.0 (log scale)
- `scale_pos_weight`: 1.0-20.0

## Visualization and Analysis

The tool automatically generates:

1. **Enhanced MLflow Artifacts**:
   - PR curves for validation and final holdout
   - ROC curves for validation and final holdout
   - **Advanced feature importance plots** with both absolute values and percentages
   - Feature importance CSV files with percentage calculations
   - **Trial-specific feature importance tracking** for every optimization trial

2. **Feature Importance Analysis**:
   - **Dual visualization**: Absolute importance values and percentage contributions
   - **Top feature metrics**: Logged to MLflow for easy comparison across trials
   - **Cumulative importance**: Shows percentage covered by top 5 and top 10 features
   - **Value labels**: Precise importance values displayed on each bar
   - **Stability tracking**: Optimization convergence analysis across trials

3. **Optuna Visualizations**:
   - Optimization history
   - Parameter importance
   - Slice plots
   - **Feature stability plots**: Show how optimization converges over trials

4. **Comprehensive Metrics Tracking**:
   - All standard classification metrics
   - Custom optimization scores
   - **Feature importance metrics**: Top features tracked as MLflow metrics
   - Trial-by-trial comparisons with feature importance evolution

## Enhanced Feature Importance Analysis

### What's New in Feature Importance:

The enhanced feature importance analysis provides comprehensive insights into model behavior:

#### **Dual Visualization Approach**
- **Left plot**: Absolute importance values for precise numerical comparison
- **Right plot**: Percentage contributions showing relative importance
- **Value labels**: Exact values displayed on each bar for accuracy

#### **Comprehensive Data Export**
```csv
feature,importance,percentage
feature_01,0.1234,15.67%
feature_05,0.0987,12.34%
...
```

#### **MLflow Integration**
- **Top feature metrics**: Automatically logged for easy comparison
- **Cumulative percentages**: Track how many features explain most variance
- **Feature names as tags**: Searchable in MLflow UI
- **Trial-specific tracking**: See how feature importance evolves

#### **Example MLflow Metrics Logged**:
```
trial_top_1_feature_importance: 0.1234
trial_top_1_feature_percentage: 15.67
trial_top_5_cumulative_percentage: 67.89
final_top_1_feature_importance: 0.1456
final_top_10_cumulative_percentage: 89.23
```

#### **Files Generated Per Trial**:
- `feature_importances_trial_N.png` - Enhanced dual plot
- `feature_importances_trial_N.csv` - Data with percentages
- `feature_importances_final.png` - Final model analysis
- `optimization_stability_modeltype.png` - Convergence analysis

### **Usage in Analysis**:
1. **Compare trials**: See which trials found similar important features
2. **Track stability**: Identify if optimization is converging on consistent features
3. **Model interpretation**: Understand what drives your model's decisions
4. **Feature selection**: Identify candidates for feature reduction

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`

2. **Memory Issues**: Reduce `--n-trials` or use smaller datasets

3. **Database Locked**: Ensure no other processes are using the Optuna database

4. **MLflow Port Conflict**: Use different port with `mlflow ui --port 5001`

### Logging:

Check `hyperparameter_optimization.log` for detailed execution logs.

## Performance Tips

1. **Start Small**: Begin with fewer trials (50-100) to validate setup
2. **Use Persistent Storage**: Always use database storage for Optuna studies
3. **Monitor Resources**: Watch memory and CPU usage during optimization
4. **Parallel Trials**: For large datasets, consider distributed Optuna
5. **Early Stopping**: Use timeout parameter for time-bounded optimization

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is available under the MIT License.
