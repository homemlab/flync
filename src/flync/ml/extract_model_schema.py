#!/usr/bin/env python3
"""
Extract schema from a trained EBM model.

This script extracts the schema (feature names, types, statistics) from
a trained EBM model and saves it to a JSON file for later validation.

Usage:
    python extract_model_schema.py \\
        --model-path path/to/model.pkl \\
        --output-path path/to/schema.json \\
        --training-data path/to/training_data.parquet \\
        --model-name "flync_ebm" \\
        --model-version "1.0.0"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.schema_extractor import extract_and_save_schema


def main():
    parser = argparse.ArgumentParser(
        description="Extract schema from trained EBM model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pickled EBM model (.pkl file)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where schema JSON will be saved"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to training data (.parquet or .csv) for computing statistics"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name identifier for the model"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Version identifier for the model"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip computing feature statistics"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if args.training_data:
        data_path = Path(args.training_data)
        if not data_path.exists():
            print(f"Error: Training data file not found: {data_path}")
            sys.exit(1)
    
    # Extract schema
    print(f"Extracting schema from: {model_path}")
    if args.training_data:
        print(f"Using training data: {args.training_data}")
    
    try:
        schema = extract_and_save_schema(
            model_path=model_path,
            output_path=args.output_path,
            training_data_path=args.training_data,
            include_stats=not args.no_stats,
            model_name=args.model_name,
            model_version=args.model_version,
        )
        
        print("\n✓ Schema extraction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during schema extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
