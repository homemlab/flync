# %%
import pandas as pd
import time
from tqdm import tqdm
import sys

# Print with flush to ensure terminal visibility
def status_print(message):
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)

status_print("Loading data...")
df_t = pd.read_parquet("../model/train_true.parquet")
df_f = pd.read_parquet("../model/train_false.parquet")

print(df_t.shape, df_f.shape)

# Concatenate the two dataframes
status_print("Concatenating datasets...")
df = pd.concat([df_t, df_f], axis=0)
df.drop(columns=["start","end"], inplace=True)

print(df.shape)


# %%
# import standard scaler
from sklearn.preprocessing import StandardScaler
status_print("Scaling data...")
# Initialize the scaler
scaler = StandardScaler()
# Fit the scaler to the data
scaler.fit(df.drop(columns=["name","y"]))
# Transform the data
df[df.columns[1:-1]] = scaler.transform(df.drop(columns=["name","y"]))
# Save the transformed data to a new parquet file
status_print("Saving scaled data...")
df.to_parquet("../model/train_scaled.parquet", index=False)
# Load the scaled data
df = pd.read_parquet("../model/train_scaled.parquet")
# Print the shape of the dataframe
print(df.shape)

# %%
# split the data into train, test and validation
from sklearn.model_selection import train_test_split
status_print("Splitting data into train, validation and test sets...")
# X and y
X = df.drop(columns=["name","y"])
y = df["y"]
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Print shapes of the dataframes
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# %%
# Import auto-sklearn
import autosklearn.classification
import sklearn.metrics as metrics
import time
from tqdm.auto import tqdm
import threading

# Set up auto-sklearn classifier
# We'll limit the time for the search to 30 minutes and set ensemble size to 50
status_print("Initializing Auto-sklearn classifier...")
total_time_seconds = 30*60  # 30 minutes
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=total_time_seconds,
    metric=autosklearn.metrics.precision,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
    per_run_time_limit=3*60,  # 3 minutes per run
    ensemble_kwargs={'ensemble_size': 50},
    memory_limit=10240,  # 10GB
    n_jobs=-1  # Use all available cores
)

# Create a progress bar that updates independently
progress_pbar = tqdm(total=total_time_seconds, desc="Training progress", unit="s")

# Function to update progress bar in a separate thread
def update_progress():
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > total_time_seconds:
            progress_pbar.update(total_time_seconds - progress_pbar.n)
            progress_pbar.close()
            break
            
        # Update the progress bar
        progress_pbar.update(1)
        
        # Try to print current status periodically
        if int(elapsed) % 300 < 5 and hasattr(automl, 'show_models'):
            try:
                status_print(f"Current models in ensemble: {len(automl.get_models_with_weights())}")
            except Exception:
                pass
                
        time.sleep(1)

# Fit the model to the training data
status_print("Starting auto-sklearn training (this may take a while)...")
start_time = time.time()

# Start the progress tracking in a separate thread
progress_thread = threading.Thread(target=update_progress, daemon=True)
progress_thread.start()

try:
    # Start the training process
    automl.fit(X_train, y_train, dataset_name="fly_gene_prediction")
    training_time = time.time() - start_time
    status_print(f"Training completed in {training_time:.2f} seconds")
except KeyboardInterrupt:
    status_print("Training interrupted by user. Continuing with evaluation of current ensemble...")
    training_time = time.time() - start_time
finally:
    # Make sure progress bar is closed properly
    if progress_thread.is_alive():
        progress_pbar.close()

# Get statistics of the training process
status_print("Auto-sklearn Statistics:")
print(automl.sprint_statistics())

# Evaluate on validation data
status_print("Evaluating on validation data...")
with tqdm(total=1, desc="Validation prediction") as pbar:
    y_val_pred = automl.predict(X_val)
    y_val_proba = automl.predict_proba(X_val)
    pbar.update(1)

val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
val_f1 = metrics.f1_score(y_val, y_val_pred)
val_precision = metrics.precision_score(y_val, y_val_pred)
val_recall = metrics.recall_score(y_val, y_val_pred)
val_roc_auc = metrics.roc_auc_score(y_val, y_val_proba[:,1])

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation ROC AUC: {val_roc_auc:.4f}")

# Evaluate on test data
status_print("Evaluating on test data...")
with tqdm(total=1, desc="Test prediction") as pbar:
    y_test_pred = automl.predict(X_test)
    y_test_proba = automl.predict_proba(X_test)
    pbar.update(1)

test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
test_f1 = metrics.f1_score(y_test, y_test_pred)
test_precision = metrics.precision_score(y_test, y_test_pred)
test_recall = metrics.recall_score(y_test, y_test_pred)
test_roc_auc = metrics.roc_auc_score(y_test, y_test_proba[:,1])

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")

# %%
# Show models in the ensemble
status_print("Analyzing ensemble...")
print("Models in the ensemble:")
for i, (weight, model) in enumerate(automl.get_models_with_weights()):
    print(f"Model {i}: Weight = {weight:.4f}, Type = {model.__class__.__name__}")

# Show feature importance (if available)
status_print("Extracting feature importances (if available)...")
try:
    feature_importances = automl.show_models()
    print("\nFeature importances:")
    print(feature_importances)
except Exception as e:
    print(f"Could not extract feature importances: {e}")

# Save the model
status_print("Saving model...")
import pickle
with tqdm(total=1, desc="Saving model") as pbar:
    with open('../model/automl_ensemble.pkl', 'wb') as f:
        pickle.dump(automl, f)
    pbar.update(1)
status_print("Model saved to '../model/automl_ensemble.pkl'")
status_print("All tasks completed successfully!")
