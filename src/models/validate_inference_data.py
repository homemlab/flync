#!/usr/bin/env python3
"""
Validate inference data against model schema.

This script validates a dataset against a model's schema without running
predictions, useful for data pipeline validation.

Usage:
    python validate_inference_data.py \\
        --schema-path path/to/schema.json \\
        --data-path path/to/inference_data.parquet \\
        --mode strict \\
        --check-ranges
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.schema_extractor import ModelSchema
from src.models.schema_validator import ValidationMode, validate_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Validate inference data against model schema"
    )
    parser.add_argument(
        "--schema-path",
        type=str,
        required=True,
        help="Path to schema JSON file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data file (.parquet or .csv)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "permissive", "coerce"],
        default="strict",
        help="Validation mode (default: strict)"
    )
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="Allow extra features in data (will be dropped)"
    )
    parser.add_argument(
        "--check-ranges",
        action="store_true",
        help="Check if values are within training data ranges"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save validated/corrected data to this path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation output"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    schema_path = Path(args.schema_path)
    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_path}")
        sys.exit(1)
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Load schema
    print(f"Loading schema from: {schema_path}")
    schema = ModelSchema.load(schema_path)
    print(f"  Expected features: {schema.n_features}")
    print(f"  Continuous: {schema.feature_types.count('continuous')}")
    print(f"  Boolean: {schema.feature_types.count('nominal')}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    if data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        data = pd.read_csv(data_path)
    else:
        print(f"Error: Unsupported file format: {data_path.suffix}")
        sys.exit(1)
    
    print(f"  Data shape: {data.shape}")
    print(f"  Data columns: {len(data.columns)}")
    
    # Map mode string to enum
    mode_map = {
        "strict": ValidationMode.STRICT,
        "permissive": ValidationMode.PERMISSIVE,
        "coerce": ValidationMode.COERCE,
    }
    validation_mode = mode_map[args.mode]
    
    # Validate
    print(f"\nValidating data (mode: {args.mode})...")
    result = validate_dataframe(
        data=data,
        schema=schema,
        mode=validation_mode,
        allow_extra_features=args.allow_extra,
        check_value_ranges=args.check_ranges,
    )
    
    # Print results
    print("\n" + "="*60)
    print(result.summary())
    print("="*60)
    
    if args.verbose:
        print("\nDetailed Issues:")
        for issue in result.issues:
            print(f"  {issue}")
    
    # Save corrected data if requested
    if args.output and result.data is not None:
        output_path = Path(args.output)
        print(f"\nSaving validated data to: {output_path}")
        
        if output_path.suffix == ".parquet":
            result.data.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            result.data.to_csv(output_path, index=False)
        else:
            print(f"Warning: Unsupported output format, saving as parquet")
            output_path = output_path.with_suffix(".parquet")
            result.data.to_parquet(output_path, index=False)
        
        print(f"  Saved: {output_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
