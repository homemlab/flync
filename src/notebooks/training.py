import argparse
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest Training Script")
    parser.add_argument('--pos', type=str, required=True, help='Path to positive label parquet file')
    parser.add_argument('--neg', type=str, required=True, help='Path to negative label parquet file')
    parser.add_argument('--drop-cols', nargs='*', default=[], help='Columns to drop before training')
    parser.add_argument('--fillna', type=str, default='mean', choices=['mean', 'median', 'most_frequent', 'constant'], help='Missing value fill strategy')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--model-path', type=str, required=True, help='Where to save the trained model')
    parser.add_argument('--n-estimators', type=int, default=100, help='RandomForest n_estimators')
    parser.add_argument('--max-depth', type=int, default=None, help='RandomForest max_depth')
    parser.add_argument('--metrics-csv', type=str, required=True, help='Path to save metrics as CSV')
    return parser.parse_args()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Random Forest training script.")

    args = parse_args()
    logger.info("Arguments parsed.")

    # Load data
    logger.info(f"Loading positive data from {args.pos}")
    df_pos = pd.read_parquet(args.pos)
    logger.info(f"Loading negative data from {args.neg}")
    df_neg = pd.read_parquet(args.neg)

    # Check if columns match
    logger.info("Checking if columns match between datasets.")
    if set(df_pos.columns) != set(df_neg.columns):
        logger.error("Columns in positive and negative datasets do not match.")
        raise ValueError("Columns in positive and negative datasets do not match.")

    # Drop columns
    logger.info(f"Dropping columns: {args.drop_cols}")
    drop_cols = set(args.drop_cols)
    X_pos = df_pos.drop(columns=drop_cols | {args.target}, errors='ignore')
    X_neg = df_neg.drop(columns=drop_cols | {args.target}, errors='ignore')
    y_pos = df_pos[args.target]
    y_neg = df_neg[args.target]

    # Drop records with any NA value after column selection, per class
    n_pos_before = len(X_pos)
    n_neg_before = len(X_neg)
    pos_na_mask = X_pos.isna().any(axis=1)
    neg_na_mask = X_neg.isna().any(axis=1)
    n_pos_dropped = pos_na_mask.sum()
    n_neg_dropped = neg_na_mask.sum()
    X_pos = X_pos[~pos_na_mask]
    y_pos = y_pos[~pos_na_mask]
    X_neg = X_neg[~neg_na_mask]
    y_neg = y_neg[~neg_na_mask]
    logger.info(
        f"Positive class: dropped {n_pos_dropped} of {n_pos_before} rows ({100.0 * n_pos_dropped / n_pos_before:.2f}%) due to NA"
    )
    logger.info(
        f"Negative class: dropped {n_neg_dropped} of {n_neg_before} rows ({100.0 * n_neg_dropped / n_neg_before:.2f}%) due to NA"
    )

    # Check for string columns
    string_cols = X_pos.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        logger.warning(f"String columns detected: {list(string_cols)}. Converting to category.")
        for col in string_cols:
            X_pos[col] = X_pos[col].astype('category')
            X_neg[col] = X_neg[col].astype('category')
    else:
        logger.info("No string columns detected.")

    # Concatenate
    logger.info("Concatenating datasets.")
    X = pd.concat([X_pos, X_neg], axis=0).reset_index(drop=True)
    y = pd.concat([y_pos, y_neg], axis=0).reset_index(drop=True)

    # Check balance
    class_counts = y.value_counts()
    logger.info(f"Class distribution:\n{class_counts}")
    is_balanced = True # np.isclose(class_counts.min() / class_counts.max(), 1.0, atol=0.2)
    logger.info(f"Is balanced: {is_balanced}")

    # Fill missing values
    logger.info(f"Filling missing values using strategy: {args.fillna}")
    imputer = SimpleImputer(strategy=args.fillna)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode categoricals
    cat_cols = X_imputed.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        logger.info(f"Encoding categorical columns: {list(cat_cols)}")
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(X_imputed[cat_cols])
        X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
        X_num = X_imputed.drop(columns=cat_cols)
        X_final = pd.concat([X_num.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)
    else:
        logger.info("No categorical columns to encode.")
        X_final = X_imputed

    # Handle imbalance
    if not is_balanced:
        logger.info("Applying SMOTE to handle class imbalance.")
        smote = SMOTE()
        X_final, y = smote.fit_resample(X_final, y)
    else:
        logger.info("No imbalance handling needed.")

    # Train/test split
    logger.info("Splitting data into train and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)

    # Train model
    logger.info("Training Random Forest model.")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    logger.info("Model training complete.")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")

    # Save metrics to CSV
    metrics = {
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "fillna": args.fillna,
        "drop_cols": ','.join(args.drop_cols)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(args.metrics_csv, index=False)
    logger.info(f"Metrics saved to {args.metrics_csv}")

    # Save model
    joblib.dump(clf, args.model_path)
    logger.info(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
