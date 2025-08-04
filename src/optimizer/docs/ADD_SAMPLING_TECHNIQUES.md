## Summary

I've successfully added SMOTE, SMOTETomek, and random undersampling capabilities to the `prepare_data.py` script. Here's what I added:

### New Features:

1. **SMOTE (Synthetic Minority Oversampling Technique)**:
   - Creates synthetic examples of minority class by interpolating between existing samples
   - Configurable k-neighbors parameter
   - Best for datasets where you want to increase minority class representation

2. **SMOTE-Tomek**:
   - Combines SMOTE oversampling with Tomek links undersampling
   - First applies SMOTE, then removes Tomek links (pairs of samples from different classes that are nearest neighbors)
   - Helps clean up the decision boundary while addressing class imbalance

3. **Random Undersampling**:
   - Randomly removes samples from the majority class
   - Simple and fast approach
   - Good when you have a large dataset and want to reduce training time

### New Command Line Arguments:

- `--sampling-strategy`: Choose between "none", "smote", "smote_tomek", "random_undersample"
- `--smote-k-neighbors`: Number of nearest neighbors for SMOTE (default: 5)
- `--smote-sampling-strategy`: Sampling strategy parameter ("auto", "minority", float ratio, etc.)

### Key Features Added:

1. **Automatic dependency checking**: Script checks if `imbalanced-learn` is installed
2. **Feature validation**: Warns about non-numeric features, missing values, and insufficient samples for SMOTE
3. **Comprehensive logging**: Shows original and resampled class distributions
4. **Error handling**: Graceful fallback if sampling fails
5. **Flexible parameter parsing**: Handles both string and numeric sampling strategies
6. **Only applies to training set**: Sampling is only applied to the training set, keeping test and holdout sets unchanged

### Usage Examples:

```bash
# Basic SMOTE oversampling
python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet \
    --output-dir ./data --sampling-strategy smote

# SMOTE with custom parameters
python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet \
    --output-dir ./data --sampling-strategy smote --smote-k-neighbors 3 \
    --smote-sampling-strategy 0.8

# SMOTE-Tomek hybrid approach
python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet \
    --output-dir ./data --sampling-strategy smote_tomek

# Random undersampling
python prepare_data.py --positive-file pos.parquet --negative-file neg.parquet \
    --output-dir ./data --sampling-strategy random_undersample
```

The script maintains backward compatibility - existing functionality works exactly as before when `--sampling-strategy none` (the default) is used.

**✅ Sampling is ONLY applied to the training set**

Looking at lines 139-145 in the code:

```python
# Apply sampling strategies to training set only
if sampling_strategy != "none":
    logger.info(f"Applying {sampling_strategy} sampling to training set...")
    train_df = _apply_sampling_strategy(
        train_df, target_column, sampling_strategy, 
        smote_k_neighbors, smote_sampling_strategy, random_state
    )
```

The key points that ensure model evaluation integrity are:

## 1. **Proper Data Split Order**
1. First, the data is split into train/test/holdout sets using stratified sampling
2. Then, sampling techniques (SMOTE, SMOTE-Tomek, Random Undersampling) are applied **ONLY** to `train_df`
3. `test_df` and `holdout_df` remain completely untouched by any sampling

## 2. **Clean Evaluation Sets**
- **Test set**: Used during hyperparameter optimization for validation - remains original
- **Holdout set**: Used for final model evaluation - remains original
- **Training set**: Only set that gets sampling applied for learning

## 3. **Why This Matters**
This is crucial because:
- **No data leakage**: Synthetic samples never contaminate evaluation
- **Realistic metrics**: Accuracy, precision, recall, F1, ROC-AUC are measured on real, unseen data
- **Honest performance**: Model performance reflects how it will perform on real-world data
- **Valid comparison**: You can compare models trained with different sampling strategies fairly

## 4. **Additional Verification**
Let me also check that the logging clearly shows this: Perfect! The logging at lines 174-176 will show you the class distributions for all three sets:

```python
# Log class distributions
for name, df in [("Train", train_df), ("Test", test_df), ("Holdout", holdout_df)]:
    dist = df[target_column].value_counts(normalize=True).sort_index()
    logger.info(f"  {name} class distribution: {dist.to_dict()}")
```

## **What You'll See in the Logs**

When you run the script with sampling, you'll see something like:

```
[INFO] Applying smote sampling to training set...
[INFO] Original training distribution: {0: 1000, 1: 100}
[INFO] Resampled training distribution: {0: 1000, 1: 1000}
[INFO] Dataset size change: 1100 -> 2000 samples
[INFO] Sampling configuration:
[INFO]   Strategy: smote
[INFO]   K-neighbors: 5
[INFO]   Sampling strategy parameter: auto
[INFO]   Train class distribution: {0: 0.5, 1: 0.5}
[INFO]   Test class distribution: {0: 0.91, 1: 0.09}    # <- Original distribution preserved!
[INFO]   Holdout class distribution: {0: 0.90, 1: 0.10} # <- Original distribution preserved!
```

Notice how only the training set shows balanced classes (0.5/0.5), while test and holdout maintain the original imbalanced distribution. This is exactly what you want for unbiased evaluation!

**Your model evaluation metrics will be completely clean and uncontaminated by the sampling techniques.** 🎯