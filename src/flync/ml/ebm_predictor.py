"""
Production-ready EBM predictor with built-in schema validation.

This module provides a wrapper around the EBM model that automatically
validates input data before making predictions.
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from .schema_extractor import ModelSchema
from .schema_validator import ValidationMode, ValidationResult, SchemaValidator


class EBMPredictor:
    """
    EBM Classifier with automatic input validation.

    This class wraps a trained EBM model and its schema, providing automatic
    validation of inference data before predictions.

    Parameters
    ----------
    model_path : str or Path
        Path to pickled EBM model
    schema_path : str or Path
        Path to schema JSON file
    validation_mode : ValidationMode, default=ValidationMode.PERMISSIVE
        Validation mode for inference data
    allow_extra_features : bool, default=True
        Whether to allow and drop extra features
    check_value_ranges : bool, default=False
        Whether to check if values are within training ranges
    auto_validate : bool, default=True
        Whether to automatically validate data before prediction

    Examples
    --------
    >>> predictor = EBMPredictor("model.pkl", "schema.json")
    >>> predictions, validation = predictor.predict(new_data)
    >>> if validation.is_valid:
    ...     print(f"Predictions: {predictions}")
    ... else:
    ...     print(f"Validation failed: {validation.summary()}")
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        schema_path: Union[str, Path],
        validation_mode: ValidationMode = ValidationMode.PERMISSIVE,
        allow_extra_features: bool = True,
        check_value_ranges: bool = False,
        auto_validate: bool = True,
    ):
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path)
        self.validation_mode = validation_mode
        self.allow_extra_features = allow_extra_features
        self.check_value_ranges = check_value_ranges
        self.auto_validate = auto_validate

        # Load model and schema
        self.model = self._load_model()
        self.schema = ModelSchema.load(schema_path)

        # Create validator
        self.validator = SchemaValidator(
            schema=self.schema,
            mode=validation_mode,
            allow_extra_features=allow_extra_features,
            check_value_ranges=check_value_ranges,
        )

    def _load_model(self) -> ExplainableBoostingClassifier:
        """Load the pickled model."""
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, ExplainableBoostingClassifier):
            raise TypeError(
                f"Expected ExplainableBoostingClassifier, got {type(model).__name__}"
            )

        return model

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data against schema without making predictions.

        Parameters
        ----------
        data : pd.DataFrame
            Data to validate

        Returns
        -------
        ValidationResult
            Validation result with issues and corrected data if applicable
        """
        return self.validator.validate(data)

    def predict(
        self,
        data: pd.DataFrame,
        validate: Optional[bool] = None,
        return_validation: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ValidationResult]]:
        """
        Make predictions on new data with optional validation.

        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction
        validate : bool, optional
            Whether to validate data. If None, uses auto_validate setting.
        return_validation : bool, default=True
            Whether to return validation result along with predictions

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels
        validation_result : ValidationResult, optional
            Validation result (only if return_validation=True)

        Raises
        ------
        ValueError
            If validation fails in strict mode
        """
        should_validate = validate if validate is not None else self.auto_validate
        validation_result = None
        validated_data = data

        if should_validate:
            validation_result = self.validate(data)

            if not validation_result.is_valid:
                if self.validation_mode == ValidationMode.STRICT:
                    raise ValueError(
                        f"Data validation failed:\n{validation_result.summary()}"
                    )
                elif validation_result.data is not None:
                    # Use corrected data
                    validated_data = validation_result.data
                else:
                    raise ValueError(
                        f"Data validation failed and could not be corrected:\n{validation_result.summary()}"
                    )
            elif validation_result.data is not None:
                # Use potentially reordered/coerced data
                validated_data = validation_result.data

        # Make predictions
        predictions = self.model.predict(validated_data)

        if return_validation and should_validate:
            return predictions, validation_result
        else:
            return predictions

    def predict_proba(
        self,
        data: pd.DataFrame,
        validate: Optional[bool] = None,
        return_validation: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ValidationResult]]:
        """
        Predict class probabilities with optional validation.

        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction
        validate : bool, optional
            Whether to validate data. If None, uses auto_validate setting.
        return_validation : bool, default=True
            Whether to return validation result along with predictions

        Returns
        -------
        probabilities : np.ndarray
            Predicted class probabilities
        validation_result : ValidationResult, optional
            Validation result (only if return_validation=True)

        Raises
        ------
        ValueError
            If validation fails in strict mode
        """
        should_validate = validate if validate is not None else self.auto_validate
        validation_result = None
        validated_data = data

        if should_validate:
            validation_result = self.validate(data)

            if not validation_result.is_valid:
                if self.validation_mode == ValidationMode.STRICT:
                    raise ValueError(
                        f"Data validation failed:\n{validation_result.summary()}"
                    )
                elif validation_result.data is not None:
                    validated_data = validation_result.data
                else:
                    raise ValueError(
                        f"Data validation failed and could not be corrected:\n{validation_result.summary()}"
                    )
            elif validation_result.data is not None:
                validated_data = validation_result.data

        # Make predictions
        probabilities = self.model.predict_proba(validated_data)

        if return_validation and should_validate:
            return probabilities, validation_result
        else:
            return probabilities

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Parameters
        ----------
        top_n : int, optional
            Return only top N features. If None, returns all features.

        Returns
        -------
        pd.DataFrame
            DataFrame with features and their importance scores
        """
        # Get global explanation
        ebm_global = self.model.explain_global()

        # Extract feature names and importances
        feature_names = ebm_global.data()["names"]
        importances = ebm_global.data()["scores"]

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df.reset_index(drop=True)

    def explain_local(self, data: pd.DataFrame, index: int = 0):
        """
        Get local explanation for a specific instance.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        index : int, default=0
            Index of instance to explain

        Returns
        -------
        explanation
            EBM local explanation object
        """
        # Validate and get corrected data if needed
        validation_result = self.validate(data)
        if validation_result.data is not None:
            data = validation_result.data

        return self.model.explain_local(data.iloc[[index]])

    @property
    def feature_names(self) -> list:
        """Get list of feature names expected by the model."""
        return self.schema.feature_names

    @property
    def n_features(self) -> int:
        """Get number of features expected by the model."""
        return self.schema.n_features

    @property
    def classes(self) -> np.ndarray:
        """Get class labels."""
        return self.model.classes_


class EBMPredictorWithScaler(EBMPredictor):
    """
    EBM Predictor with optional feature scaling.

    Some preprocessing pipelines include a scaler. This class supports
    loading and applying a scaler before prediction.

    Parameters
    ----------
    model_path : str or Path
        Path to pickled EBM model
    schema_path : str or Path
        Path to schema JSON file
    scaler_path : str or Path, optional
        Path to pickled scaler (e.g., StandardScaler, MinMaxScaler)
    **kwargs
        Additional arguments passed to EBMPredictor
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        schema_path: Union[str, Path],
        scaler_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(model_path, schema_path, **kwargs)

        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.scaler = self._load_scaler(self.scaler_path) if scaler_path else None

    @classmethod
    def _load_scaler(cls, scaler_path: Union[str, Path]) -> Any:
        """Load scaler from file."""
        scaler_path = Path(scaler_path)

        with open(scaler_path, "rb") as f:
            scaler_obj = pickle.load(f)

        # Handle case where scaler is wrapped in a dict (legacy format)
        if isinstance(scaler_obj, dict) and "scaler" in scaler_obj:
            scaler = scaler_obj["scaler"]
        else:
            scaler = scaler_obj

        # Validate that we got a proper scaler object
        if not hasattr(scaler, "transform"):
            raise ValueError(
                f"Loaded scaler from {scaler_path} doesn't have a 'transform' method. "
                f"Got type: {type(scaler)}. Expected a sklearn scaler object."
            )

        return scaler

    def get_continuous_features(self):
        """Get list of continuous feature names from schema."""
        continuous_features = [
            feat
            for i, feat in enumerate(self.schema.feature_names)
            if self.schema.feature_types[i] == "continuous"
        ]
        return continuous_features

    def predict(self, data, validate=None, return_validation=True, **kwargs):
        """Predict with scaling of continuous features."""
        # Validate using parent's validate method
        should_validate = validate if validate is not None else self.auto_validate
        validation_result = None
        validated_data = data

        if should_validate:
            validation_result = self.validate(data)

            if not validation_result.is_valid:
                if self.validation_mode == ValidationMode.STRICT:
                    raise ValueError(
                        f"Data validation failed:\n{validation_result.summary()}"
                    )
                elif validation_result.data is not None:
                    # Use corrected data
                    validated_data = validation_result.data
                else:
                    raise ValueError(
                        f"Data validation failed and could not be corrected:\n{validation_result.summary()}"
                    )
            elif validation_result.data is not None:
                # Use potentially reordered/coerced data
                validated_data = validation_result.data

        # Scale continuous features if scaler exists
        if self.scaler is not None:
            # Check if scaler is a dict (improperly loaded) vs proper scaler object
            if isinstance(self.scaler, dict):
                raise ValueError(
                    "Scaler was loaded as a dictionary instead of a scaler object. "
                    "This may indicate the scaler wasn't properly saved or loaded. "
                    "Please re-train the model or verify the scaler file."
                )

            continuous_features = self.get_continuous_features()
            if continuous_features:
                scaled_data = validated_data.copy()
                scaled_data[continuous_features] = self.scaler.transform(
                    validated_data[continuous_features]
                )
                validated_data = scaled_data

        # Make predictions directly (don't call super().predict to avoid double validation)
        predictions = self.model.predict(validated_data)

        if return_validation:
            return predictions, validation_result
        else:
            return predictions

    def predict_proba(
        self, data: pd.DataFrame, validate=None, return_validation=True, **kwargs
    ):
        """Predict probabilities with optional scaling."""
        # Validate using parent's validate method (same as predict)
        should_validate = validate if validate is not None else self.auto_validate
        validation_result = None
        validated_data = data

        if should_validate:
            validation_result = self.validate(data)

            if not validation_result.is_valid:
                if self.validation_mode == ValidationMode.STRICT:
                    raise ValueError(
                        f"Data validation failed:\n{validation_result.summary()}"
                    )
                elif validation_result.data is not None:
                    # Use corrected data
                    validated_data = validation_result.data
                else:
                    raise ValueError(
                        f"Data validation failed and could not be corrected:\n{validation_result.summary()}"
                    )
            elif validation_result.data is not None:
                # Use potentially reordered/coerced data
                validated_data = validation_result.data

        # Scale continuous features if scaler exists
        if self.scaler is not None:
            if isinstance(self.scaler, dict):
                raise ValueError(
                    "Scaler was loaded as a dictionary instead of a scaler object. "
                    "This may indicate the scaler wasn't properly saved or loaded. "
                    "Please re-train the model or verify the scaler file."
                )

            continuous_features = self.get_continuous_features()
            if continuous_features:
                scaled_data = validated_data.copy()
                scaled_data[continuous_features] = self.scaler.transform(
                    validated_data[continuous_features]
                )
                validated_data = scaled_data

        # Get probabilities directly (don't call parent's predict_proba to avoid double validation)
        probabilities = self.model.predict_proba(validated_data)

        if return_validation:
            return probabilities, validation_result
        else:
            return probabilities
