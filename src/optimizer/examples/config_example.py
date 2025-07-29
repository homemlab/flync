# Example Configuration for Hyperparameter Optimization

# Dataset paths (update these to point to your actual data files)
TRAIN_DATA = "./datasets/train.parquet"
TEST_DATA = "./datasets/test.parquet"
HOLDOUT_DATA = "./datasets/holdout.parquet"
TARGET_COLUMN = "y"

# Model configuration
MODEL_TYPE = "randomforest"  # or "xgboost"
OPTIMIZATION_METRICS = ["precision", "f1"]  # metrics to optimize (will be averaged)
OPTIMIZATION_DIRECTION = "maximize"  # or "minimize"

# Optuna configuration
STUDY_NAME = "rf_precision_f1_study_v1"
STORAGE_URL = "sqlite:///optuna_study.db"
N_TRIALS = 100
TIMEOUT = 3600  # seconds (1 hour)

# MLflow configuration
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "RandomForest_Precision_F1_Optimization"

# Other settings
RANDOM_STATE = 42
PROJECT_NAME = "ML Model Optimization"
DATASET_VERSION = "v1.0"
