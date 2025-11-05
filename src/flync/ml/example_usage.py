#!/usr/bin/env python3
"""
Example usage of the EBM predictor with schema validation.

This script demonstrates how to:
1. Extract schema from a trained model
2. Use the predictor with automatic validation
3. Handle validation results
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    EBMPredictor,
    ValidationMode,
    extract_and_save_schema,
)


def example_1_extract_schema():
    """Example 1: Extract schema from trained model."""
    print("="*60)
    print("Example 1: Extracting Schema from Model")
    print("="*60)
    
    model_path = "/home/chlab/flync/final_train_artifacts/FINAL_flync_ebm_model_gffutils.pkl"
    training_data_path = "/home/chlab/flync/final_train_artifacts/X_train_final.parquet"
    schema_path = "/home/chlab/flync/final_train_artifacts/flync_model_schema.json"
    
    print(f"\nExtracting schema from: {model_path}")
    print(f"Using training data: {training_data_path}")
    
    schema = extract_and_save_schema(
        model_path=model_path,
        output_path=schema_path,
        training_data_path=training_data_path,
        include_stats=True,
        model_name="flync_ebm",
        model_version="1.0.0",
    )
    
    print(f"\n✓ Schema saved to: {schema_path}")
    return schema_path


def example_2_predict_with_validation(schema_path):
    """Example 2: Make predictions with automatic validation."""
    print("\n" + "="*60)
    print("Example 2: Prediction with Automatic Validation")
    print("="*60)
    
    model_path = "/home/chlab/flync/final_train_artifacts/FINAL_flync_ebm_model_gffutils.pkl"
    test_data_path = "/home/chlab/flync/final_train_artifacts/X_test_final.parquet"
    
    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    test_data = pd.read_parquet(test_data_path)
    
    # Remove target if present
    if 'y' in test_data.columns:
        y_true = test_data['y']
        test_data = test_data.drop(columns=['y'])
    
    print(f"Test data shape: {test_data.shape}")
    
    # Create predictor with permissive mode (auto-fixes reordering issues)
    print(f"\nCreating predictor (mode: PERMISSIVE)...")
    predictor = EBMPredictor(
        model_path=model_path,
        schema_path=schema_path,
        validation_mode=ValidationMode.PERMISSIVE,
        allow_extra_features=True,
        check_value_ranges=False,
    )
    
    print(f"Predictor loaded successfully!")
    print(f"  Expected features: {predictor.n_features}")
    print(f"  Classes: {predictor.classes}")
    
    # Make predictions with validation
    print(f"\nMaking predictions on {len(test_data)} samples...")
    predictions, validation_result = predictor.predict(test_data)
    
    print(f"\n✓ Predictions completed!")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Unique predictions: {set(predictions)}")
    
    # Show validation summary
    print(f"\nValidation Summary:")
    print(f"  Valid: {validation_result.is_valid}")
    print(f"  Errors: {len(validation_result.errors)}")
    print(f"  Warnings: {len(validation_result.warnings)}")
    print(f"  Info: {len(validation_result.infos)}")
    
    if validation_result.issues:
        print(f"\nValidation Issues:")
        for issue in validation_result.issues[:5]:  # Show first 5
            print(f"  - {issue}")
    
    # Get probabilities
    print(f"\nGetting prediction probabilities...")
    probas, _ = predictor.predict_proba(test_data)
    print(f"  Probabilities shape: {probas.shape}")
    print(f"  Sample probabilities (first 3):")
    for i in range(min(3, len(probas))):
        print(f"    {i}: {probas[i]}")
    
    return predictor


def example_3_validation_modes(schema_path):
    """Example 3: Demonstrate different validation modes."""
    print("\n" + "="*60)
    print("Example 3: Different Validation Modes")
    print("="*60)
    
    model_path = "/home/chlab/flync/final_train_artifacts/FINAL_flync_ebm_model_gffutils.pkl"
    test_data_path = "/home/chlab/flync/final_train_artifacts/X_test_final.parquet"
    
    # Load and shuffle columns to simulate wrong order
    test_data = pd.read_parquet(test_data_path)
    if 'y' in test_data.columns:
        test_data = test_data.drop(columns=['y'])
    
    # Shuffle columns
    import numpy as np
    cols = list(test_data.columns)
    np.random.seed(42)
    np.random.shuffle(cols)
    shuffled_data = test_data[cols]
    
    print(f"\nCreated shuffled test data (wrong column order)")
    print(f"  Original order (first 5): {list(test_data.columns[:5])}")
    print(f"  Shuffled order (first 5): {list(shuffled_data.columns[:5])}")
    
    # Mode 1: STRICT - Should fail
    print(f"\n1. STRICT Mode (should fail on wrong order):")
    try:
        predictor_strict = EBMPredictor(
            model_path=model_path,
            schema_path=schema_path,
            validation_mode=ValidationMode.STRICT,
        )
        predictions, result = predictor_strict.predict(shuffled_data)
        print(f"   Validation passed: {result.is_valid}")
    except ValueError as e:
        print(f"   ✓ Failed as expected: {str(e)[:100]}...")
    
    # Mode 2: PERMISSIVE - Should auto-fix
    print(f"\n2. PERMISSIVE Mode (should auto-fix order):")
    predictor_permissive = EBMPredictor(
        model_path=model_path,
        schema_path=schema_path,
        validation_mode=ValidationMode.PERMISSIVE,
    )
    predictions, result = predictor_permissive.predict(shuffled_data)
    print(f"   ✓ Validation passed: {result.is_valid}")
    print(f"   Predictions made: {len(predictions)}")
    print(f"   Issues: {len(result.issues)} (auto-fixed)")


def example_4_feature_importance(predictor):
    """Example 4: Get feature importance."""
    print("\n" + "="*60)
    print("Example 4: Feature Importance")
    print("="*60)
    
    print(f"\nGetting top 10 most important features...")
    importance_df = predictor.get_feature_importance(top_n=10)
    
    print(f"\nTop 10 Features:")
    print(importance_df.to_string(index=True))


def main():
    """Run all examples."""
    print("\n" + "🚀 "*20)
    print("EBM Schema Validation - Example Usage")
    print("🚀 "*20)
    
    try:
        # Example 1: Extract schema
        schema_path = example_1_extract_schema()
        
        # Example 2: Predict with validation
        predictor = example_2_predict_with_validation(schema_path)
        
        # Example 3: Different validation modes
        example_3_validation_modes(schema_path)
        
        # Example 4: Feature importance
        example_4_feature_importance(predictor)
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
