"""
Schema extraction utilities for EBM model validation.

This module provides functionality to extract and persist the schema
of a trained EBM model, including feature names, types, and statistics.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier


class ModelSchema:
    """Container for model schema information."""
    
    def __init__(
        self,
        feature_names: List[str],
        feature_types: List[str],
        feature_dtypes: Dict[str, str],
        n_features: int,
        metadata: Optional[Dict[str, Any]] = None,
        feature_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.feature_dtypes = feature_dtypes
        self.n_features = n_features
        self.metadata = metadata or {}
        self.feature_stats = feature_stats or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "feature_dtypes": self.feature_dtypes,
            "n_features": self.n_features,
            "metadata": self.metadata,
            "feature_stats": self.feature_stats,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSchema":
        """Create schema from dictionary."""
        return cls(
            feature_names=data["feature_names"],
            feature_types=data["feature_types"],
            feature_dtypes=data["feature_dtypes"],
            n_features=data["n_features"],
            metadata=data.get("metadata", {}),
            feature_stats=data.get("feature_stats", {}),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save schema to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelSchema":
        """Load schema from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def extract_schema_from_model(
    model: ExplainableBoostingClassifier,
    training_data: Optional[pd.DataFrame] = None,
    include_stats: bool = True,
    compute_percentiles: bool = True,
    model_version: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ModelSchema:
    """
    Extract schema from a trained EBM model.
    
    Parameters
    ----------
    model : ExplainableBoostingClassifier
        Trained EBM model
    training_data : pd.DataFrame, optional
        Training data used to fit the model. If provided, will compute
        statistics for validation ranges.
    include_stats : bool, default=True
        Whether to compute feature statistics from training data
    compute_percentiles : bool, default=True
        Whether to compute percentiles (5th, 25th, 75th, 95th) for continuous features
    model_version : str, optional
        Version identifier for the model
    model_name : str, optional
        Name identifier for the model
    
    Returns
    -------
    ModelSchema
        Schema object containing all model feature requirements
    """
    # Extract basic feature information
    feature_names = list(model.feature_names_in_)
    feature_types = list(model.feature_types_in_)
    n_features = model.n_features_in_
    
    # Initialize dtypes dictionary
    feature_dtypes = {}
    
    # Determine expected dtypes based on feature types
    for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
        if ftype == "continuous":
            feature_dtypes[name] = "float64"
        elif ftype == "nominal":
            # For EBM, nominal features that are boolean should be bool type
            feature_dtypes[name] = "bool"
        else:
            feature_dtypes[name] = "object"
    
    # Override with actual dtypes from training data if provided
    if training_data is not None:
        for col in feature_names:
            if col in training_data.columns:
                feature_dtypes[col] = str(training_data[col].dtype)
    
    # Compute feature statistics
    feature_stats = {}
    if include_stats and training_data is not None:
        for i, name in enumerate(feature_names):
            if name not in training_data.columns:
                continue
            
            col = training_data[name]
            ftype = feature_types[i]
            
            stats = {
                "feature_type": ftype,
                "null_count": int(col.isna().sum()),
                "null_percentage": float(col.isna().mean() * 100),
            }
            
            if ftype == "continuous":
                # Statistics for continuous features
                stats.update({
                    "min": float(col.min()) if not col.isna().all() else None,
                    "max": float(col.max()) if not col.isna().all() else None,
                    "mean": float(col.mean()) if not col.isna().all() else None,
                    "std": float(col.std()) if not col.isna().all() else None,
                    "median": float(col.median()) if not col.isna().all() else None,
                })
                
                if compute_percentiles and not col.isna().all():
                    percentiles = col.quantile([0.05, 0.25, 0.75, 0.95])
                    stats.update({
                        "p05": float(percentiles.iloc[0]),
                        "p25": float(percentiles.iloc[1]),
                        "p75": float(percentiles.iloc[2]),
                        "p95": float(percentiles.iloc[3]),
                    })
            
            elif ftype == "nominal":
                # Statistics for boolean features
                value_counts = col.value_counts()
                stats.update({
                    "unique_values": [bool(v) for v in col.unique() if pd.notna(v)],
                    "value_counts": {str(k): int(v) for k, v in value_counts.items()},
                    "true_percentage": float(col.sum() / len(col) * 100) if len(col) > 0 else 0.0,
                })
            
            feature_stats[name] = stats
    
    # Create metadata
    metadata = {
        "extraction_date": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "model_version": model_version,
        "model_name": model_name,
        "sklearn_version": None,  # Could add sklearn.__version__ if needed
    }
    
    return ModelSchema(
        feature_names=feature_names,
        feature_types=feature_types,
        feature_dtypes=feature_dtypes,
        n_features=n_features,
        metadata=metadata,
        feature_stats=feature_stats,
    )


def extract_and_save_schema(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    training_data_path: Optional[Union[str, Path]] = None,
    include_stats: bool = True,
    model_version: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ModelSchema:
    """
    Load model from file, extract schema, and save to JSON.
    
    Parameters
    ----------
    model_path : str or Path
        Path to pickled model file
    output_path : str or Path
        Path where schema JSON will be saved
    training_data_path : str or Path, optional
        Path to training data (parquet or csv). If provided, will compute statistics.
    include_stats : bool, default=True
        Whether to compute feature statistics
    model_version : str, optional
        Version identifier for the model
    model_name : str, optional
        Name identifier for the model
    
    Returns
    -------
    ModelSchema
        Extracted schema object
    """
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load training data if provided
    training_data = None
    if training_data_path is not None:
        data_path = Path(training_data_path)
        if data_path.suffix == ".parquet":
            training_data = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            training_data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Remove target column if present
        if "y" in training_data.columns:
            training_data = training_data.drop(columns=["y"])
    
    # Extract schema
    schema = extract_schema_from_model(
        model=model,
        training_data=training_data,
        include_stats=include_stats,
        model_version=model_version,
        model_name=model_name,
    )
    
    # Save schema
    schema.save(output_path)
    
    print(f"Schema extracted successfully!")
    print(f"  Features: {schema.n_features}")
    print(f"  Continuous: {schema.feature_types.count('continuous')}")
    print(f"  Boolean: {schema.feature_types.count('nominal')}")
    print(f"  Saved to: {output_path}")
    
    return schema
