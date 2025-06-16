"""Machine learning components for FLYNC."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from ..utils import LoggerMixin
from ..data import get_model_path, get_training_data_path


class LncRNAClassifier(LoggerMixin):
    """Random Forest classifier for lncRNA prediction."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_estimators: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        self.model_path = model_path
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> None:
        """Load a pre-trained model."""
        self.logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', [])
        else:
            # Legacy format - just the model
            self.model = model_data
        
        self.is_fitted = True
        self.logger.info("Model loaded successfully")
    
    def save_model(self, model_path: Path) -> None:
        """Save the trained model."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction of data to use for testing
            save_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting model training")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure features are in correct order
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
            
            # Reorder columns to match training
            available_features = [f for f in self.feature_names if f in X.columns]
            X = X[available_features]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'lncRNA': predictions,
            'Prob_False': probabilities[:, 0],
            'Prob_True': probabilities[:, 1]
        })
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class FeatureProcessor(LoggerMixin):
    """Process and prepare features for ML model."""
    
    @staticmethod
    def load_feature_table(feature_table_path: Path) -> pd.DataFrame:
        """Load feature table from CSV."""
        return pd.read_csv(feature_table_path)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input.
        
        Args:
            df: Raw feature dataframe
            
        Returns:
            Processed feature dataframe
        """
        self.logger.info("Preparing features for ML model")
        
        # Remove features that were dropped in the original model
        features_to_drop = ['cov_me3', 'mean_pPcons27']
        for feature in features_to_drop:
            if feature in df.columns:
                df = df.drop(feature, axis=1)
                self.logger.debug(f"Dropped feature: {feature}")
        
        # Select feature columns (exclude metadata columns)
        feature_start_col = 'length'
        feature_end_col = 'mean_pPcons124'
        
        if feature_start_col in df.columns and feature_end_col in df.columns:
            start_idx = df.columns.get_loc(feature_start_col)
            end_idx = df.columns.get_loc(feature_end_col)
            feature_cols = df.columns[start_idx:end_idx + 1]
            feature_df = df[feature_cols].copy()
        else:
            # Fall back to all numeric columns
            feature_df = df.select_dtypes(include=[np.number]).copy()
        
        self.logger.info(f"Prepared {len(feature_df.columns)} features for {len(feature_df)} samples")
        
        return feature_df
    
    def create_training_data(
        self,
        positive_samples: Path,
        negative_samples: Path
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training data from positive and negative sample files.
        
        Args:
            positive_samples: Path to positive sample features
            negative_samples: Path to negative sample features
            
        Returns:
            Tuple of (features, labels)
        """
        self.logger.info("Creating training data")
        
        # Load positive samples
        pos_df = self.load_feature_table(positive_samples)
        pos_df['label'] = 1
        
        # Load negative samples
        neg_df = self.load_feature_table(negative_samples)
        neg_df['label'] = 0
        
        # Combine datasets
        combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
        
        # Prepare features
        X = self.prepare_features(combined_df)
        y = combined_df['label']
        
        self.logger.info(f"Created training data: {len(pos_df)} positive, {len(neg_df)} negative samples")
        
        return X, y


def retrain_model(
    positive_samples_path: Path,
    negative_samples_path: Path,
    output_model_path: Path,
    **model_kwargs
) -> Dict[str, float]:
    """Retrain the lncRNA classifier with new data.
    
    Args:
        positive_samples_path: Path to positive sample features
        negative_samples_path: Path to negative sample features  
        output_model_path: Path to save the retrained model
        **model_kwargs: Additional arguments for RandomForestClassifier
        
    Returns:
        Dictionary with training metrics
    """
    processor = FeatureProcessor()
    X, y = processor.create_training_data(positive_samples_path, negative_samples_path)
    
    classifier = LncRNAClassifier(**model_kwargs)
    metrics = classifier.train(X, y, save_path=output_model_path)
    
    return metrics
