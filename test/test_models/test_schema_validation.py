"""
Unit tests for schema validation system.

Run with: pytest test/test_models/test_schema_validation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import (
    EBMPredictor,
    ModelSchema,
    ValidationLevel,
    ValidationMode,
    validate_dataframe,
)


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return ModelSchema(
        feature_names=["feat1", "feat2", "feat3", "is_true", "is_false"],
        feature_types=["continuous", "continuous", "continuous", "nominal", "nominal"],
        feature_dtypes={
            "feat1": "float64",
            "feat2": "float64",
            "feat3": "float64",
            "is_true": "bool",
            "is_false": "bool",
        },
        n_features=5,
        metadata={"test": True},
        feature_stats={
            "feat1": {
                "feature_type": "continuous",
                "min": 0.0,
                "max": 10.0,
                "mean": 5.0,
            },
            "is_true": {
                "feature_type": "nominal",
                "unique_values": [True, False],
            },
        },
    )


@pytest.fixture
def valid_data():
    """Create valid test data."""
    return pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [4.0, 5.0, 6.0],
        "feat3": [7.0, 8.0, 9.0],
        "is_true": [True, False, True],
        "is_false": [False, True, False],
    })


class TestSchemaExtraction:
    """Tests for schema extraction."""
    
    def test_schema_to_dict(self, sample_schema):
        """Test schema serialization to dict."""
        schema_dict = sample_schema.to_dict()
        
        assert "feature_names" in schema_dict
        assert "feature_types" in schema_dict
        assert "n_features" in schema_dict
        assert schema_dict["n_features"] == 5
    
    def test_schema_round_trip(self, sample_schema, tmp_path):
        """Test saving and loading schema."""
        schema_path = tmp_path / "schema.json"
        
        # Save
        sample_schema.save(schema_path)
        assert schema_path.exists()
        
        # Load
        loaded_schema = ModelSchema.load(schema_path)
        
        assert loaded_schema.feature_names == sample_schema.feature_names
        assert loaded_schema.feature_types == sample_schema.feature_types
        assert loaded_schema.n_features == sample_schema.n_features


class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_valid_data(self, sample_schema, valid_data):
        """Test validation passes for valid data."""
        result = validate_dataframe(
            valid_data, sample_schema, mode=ValidationMode.STRICT
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_missing_feature(self, sample_schema, valid_data):
        """Test detection of missing features."""
        # Remove a feature
        data_missing = valid_data.drop(columns=["feat1"])
        
        result = validate_dataframe(
            data_missing, sample_schema, mode=ValidationMode.STRICT
        )
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("Missing required feature" in str(e) for e in result.errors)
    
    def test_extra_feature_strict(self, sample_schema, valid_data):
        """Test extra feature detection in strict mode."""
        # Add an extra feature
        data_extra = valid_data.copy()
        data_extra["extra_col"] = [1, 2, 3]
        
        result = validate_dataframe(
            data_extra, sample_schema, mode=ValidationMode.STRICT
        )
        
        assert not result.is_valid
        assert any("Extra feature" in str(e) for e in result.errors)
    
    def test_extra_feature_permissive(self, sample_schema, valid_data):
        """Test extra feature handling in permissive mode."""
        # Add an extra feature
        data_extra = valid_data.copy()
        data_extra["extra_col"] = [1, 2, 3]
        
        result = validate_dataframe(
            data_extra,
            sample_schema,
            mode=ValidationMode.PERMISSIVE,
            allow_extra_features=True,
        )
        
        # Should be valid with warning
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "extra_col" not in result.data.columns
    
    def test_wrong_order(self, sample_schema, valid_data):
        """Test feature order validation."""
        # Shuffle columns
        data_shuffled = valid_data[["feat3", "feat1", "feat2", "is_false", "is_true"]]
        
        # Strict mode should fail
        result_strict = validate_dataframe(
            data_shuffled, sample_schema, mode=ValidationMode.STRICT
        )
        assert not result_strict.is_valid
        
        # Permissive mode should auto-fix
        result_permissive = validate_dataframe(
            data_shuffled, sample_schema, mode=ValidationMode.PERMISSIVE
        )
        assert result_permissive.is_valid
        assert list(result_permissive.data.columns) == sample_schema.feature_names
    
    def test_wrong_dtype(self, sample_schema, valid_data):
        """Test data type validation."""
        # Change dtype
        data_wrong_type = valid_data.copy()
        data_wrong_type["feat1"] = data_wrong_type["feat1"].astype(int)
        
        # Strict mode should fail
        result_strict = validate_dataframe(
            data_wrong_type, sample_schema, mode=ValidationMode.STRICT
        )
        assert not result_strict.is_valid
        
        # Coerce mode should fix
        result_coerce = validate_dataframe(
            data_wrong_type, sample_schema, mode=ValidationMode.COERCE
        )
        assert result_coerce.is_valid
        assert result_coerce.data["feat1"].dtype == np.float64
    
    def test_null_values(self, sample_schema, valid_data):
        """Test null value detection."""
        data_with_nulls = valid_data.copy()
        data_with_nulls.loc[0, "feat1"] = np.nan
        
        result = validate_dataframe(
            data_with_nulls, sample_schema, mode=ValidationMode.PERMISSIVE
        )
        
        # Should have warnings about nulls
        assert len(result.warnings) > 0
        assert any("null" in str(w).lower() for w in result.warnings)
    
    def test_invalid_boolean_values(self, sample_schema, valid_data):
        """Test boolean feature validation."""
        data_invalid_bool = valid_data.copy()
        data_invalid_bool["is_true"] = ["yes", "no", "maybe"]
        
        result = validate_dataframe(
            data_invalid_bool, sample_schema, mode=ValidationMode.STRICT
        )
        
        assert not result.is_valid
        assert any("Boolean feature" in str(e) for e in result.errors)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_summary(self, sample_schema, valid_data):
        """Test validation result summary."""
        result = validate_dataframe(valid_data, sample_schema)
        summary = result.summary()
        
        assert "Validation Result:" in summary
        assert "Errors:" in summary
        assert "Warnings:" in summary
    
    def test_filter_by_level(self, sample_schema, valid_data):
        """Test filtering issues by level."""
        # Create data with multiple issue types
        data_mixed = valid_data.copy()
        data_mixed["extra"] = [1, 2, 3]
        data_mixed.loc[0, "feat1"] = np.nan
        
        result = validate_dataframe(
            data_mixed,
            sample_schema,
            mode=ValidationMode.PERMISSIVE,
            allow_extra_features=True,
        )
        
        # Should have different levels
        assert len(result.errors) == 0  # No errors in permissive with allow_extra
        assert len(result.warnings) > 0  # Null values
        assert len(result.infos) > 0  # Extra feature dropped


class TestRealModel:
    """Tests using the actual trained model."""
    
    @pytest.fixture
    def model_path(self):
        return Path("/home/chlab/flync/final_train_artifacts/FINAL_flync_ebm_model_gffutils.pkl")
    
    @pytest.fixture
    def schema_path(self):
        return Path("/home/chlab/flync/final_train_artifacts/flync_model_schema.json")
    
    @pytest.fixture
    def test_data_path(self):
        return Path("/home/chlab/flync/final_train_artifacts/X_test_final.parquet")
    
    def test_real_model_prediction(self, model_path, schema_path, test_data_path):
        """Test prediction with real model."""
        if not model_path.exists() or not schema_path.exists() or not test_data_path.exists():
            pytest.skip("Real model files not available")
        
        # Load test data
        test_data = pd.read_parquet(test_data_path)
        if "y" in test_data.columns:
            test_data = test_data.drop(columns=["y"])
        
        # Create predictor
        predictor = EBMPredictor(
            model_path=model_path,
            schema_path=schema_path,
            validation_mode=ValidationMode.PERMISSIVE,
        )
        
        # Make predictions
        predictions, validation = predictor.predict(test_data.head(100))
        
        assert validation.is_valid
        assert len(predictions) == 100
        assert set(predictions).issubset({True, False})
    
    def test_real_model_feature_count(self, schema_path):
        """Test that schema has correct number of features."""
        if not schema_path.exists():
            pytest.skip("Schema file not available")
        
        schema = ModelSchema.load(schema_path)
        
        assert schema.n_features == 731
        assert schema.feature_types.count("continuous") == 36
        assert schema.feature_types.count("nominal") == 695


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
