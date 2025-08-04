# Batch Hyperparameter Optimization with YAML Configuration

This system allows you to run batch hyperparameter optimization jobs using either command-line arguments or YAML configuration files.

## Quick Start

### Using YAML Configuration (Recommended)

1. **Create a configuration file** (see `batch_optimization.yaml` for example):

```bash
# Run with config file
python src/optimizer/batch_optimization.py --config batch_optimization.yaml --dry-run
```

2. **Run the actual optimization**:

```bash
# Remove --dry-run to execute
python src/optimizer/batch_optimization.py --config batch_optimization.yaml
```

### Using Command Line Arguments (Legacy)

```bash
python src/optimizer/batch_optimization.py \
    --train-data model/train.parquet \
    --test-data model/test.parquet \
    --holdout-data model/holdout.parquet \
    --target-column y \
    --dataset-suffix my_experiment \
    --models randomforest xgboost \
    --metrics precision f1 \
    --dry-run
```

## YAML Configuration Structure

The YAML configuration file has the following main sections:

### 1. Global Defaults
```yaml
global_defaults:
  optimization_direction: "maximize"
  n_trials: 100
  random_state: 99
  # ... other defaults
```

### 2. Storage Configuration
```yaml
storage:
  mlflow_uri: "sqlite:///mlflow.db"
  storage_url: "sqlite:///optuna.db"
```

### 3. Dataset Configurations
Each dataset configuration represents a different experiment:

```yaml
dataset_configs:
  my_experiment:
    name: "My ML Experiment"
    description: "Description of the experiment"
    
    # Dataset paths (required)
    train_data: "path/to/train.parquet"
    test_data: "path/to/test.parquet"
    holdout_data: "path/to/holdout.parquet"
    target_column: "target_column_name"
    dataset_suffix: "experiment_suffix"
    
    # Experiment settings
    experiment_name: "my_mlflow_experiment"
    n_trials: 150
    random_state: 42
    
    # Feature selection (choose one)
    analyze_correlations: true
    correlation_threshold: 0.95
    # OR
    # drop_features_file: "features_to_drop.txt"
    
    # Models and metrics to test
    models: ["randomforest", "xgboost", "lightgbm"]
    metrics:
      - ["precision"]
      - ["f1"]
      - ["roc_auc"]
      - ["precision", "f1"]
    
    # Tags for experiment tracking
    tags:
      dataset_type: "genomics"
      experiment_batch: "batch_001"
      researcher: "your_name"
```

## Key Features

### 1. Multiple Dataset Support
- Configure multiple datasets in a single YAML file
- Each dataset can have different settings, models, and metrics
- Each generates its own set of optimization jobs

### 2. Flexible Configuration
- **Global defaults**: Set common parameters once
- **Per-dataset overrides**: Customize settings for specific datasets
- **Feature selection**: Choose correlation analysis or file-based feature dropping
- **Model selection**: Choose which models to run for each dataset
- **Metric combinations**: Test different metric optimization strategies

### 3. Comprehensive Tagging
- Tags are automatically passed to every trial
- Supports both predefined and custom tags
- Helps with experiment organization and filtering in MLflow

### 4. Command Line Overrides
Even when using YAML config, you can override certain settings:

```bash
python src/optimizer/batch_optimization.py \
    --config batch_optimization.yaml \
    --dry-run \
    --debug \
    --delay-between-jobs 60
```

## Example Usage Scenarios

### Scenario 1: Compare Preprocessing Methods
```yaml
dataset_configs:
  scaled_data:
    train_data: "data/train_scaled.parquet"
    # ... other settings
    tags:
      preprocessing: "scaled"
  
  raw_data:
    train_data: "data/train_raw.parquet" 
    # ... other settings
    tags:
      preprocessing: "raw"
```

### Scenario 2: Different Models for Different Datasets
```yaml
dataset_configs:
  small_dataset:
    # ... dataset settings
    models: ["randomforest", "xgboost"]  # Skip slow models
    n_trials: 50
  
  large_dataset:
    # ... dataset settings  
    models: ["randomforest", "xgboost", "lightgbm", "ebm"]  # All models
    n_trials: 200
```

### Scenario 3: A/B Testing Experiments
```yaml
dataset_configs:
  experiment_a:
    # ... settings
    analyze_correlations: true
    tags:
      variant: "feature_selection"
      
  experiment_b:
    # ... settings
    # No feature selection
    tags:
      variant: "all_features"
```

## Generated Output

For each dataset configuration and model/metric combination, the system generates:

- **Unique study names**: `{model}_optimization_{dataset_suffix}_{metrics}`
- **Unique experiment names**: `{experiment_name}_{model}_{metrics}`
- **Comprehensive tags**: All specified tags are passed to every trial
- **MLflow tracking**: Full experiment tracking with datasets, parameters, and results

## Tips

1. **Start with dry-run**: Always test with `--dry-run` first
2. **Use descriptive names**: Choose clear `dataset_suffix` and experiment names
3. **Tag everything**: Use tags to organize and filter experiments
4. **Gradual rollout**: Start with debug datasets (few trials) before full experiments
5. **Monitor resources**: Consider `job_timeout` and `delay_between_jobs` for resource management

## Command Reference

```bash
# View all options
python src/optimizer/batch_optimization.py --help

# Test configuration
python src/optimizer/batch_optimization.py --config config.yaml --dry-run --debug

# Run actual optimization
python src/optimizer/batch_optimization.py --config config.yaml

# Run with overrides
python src/optimizer/batch_optimization.py --config config.yaml --delay-between-jobs 60
```
