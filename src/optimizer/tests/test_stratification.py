#!/usr/bin/env python3
"""
Test script for stratification features in prepare_data.py

This script validates the new stratification capabilities including:
- Custom stratification columns
- Group-aware splits
- Class balancing
- Imbalance ratio control
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from collections import Counter

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_data import prepare_datasets


def create_test_data():
    """Create synthetic test data with known characteristics"""
    np.random.seed(42)
    
    # Create imbalanced dataset with groups
    n_samples = 1000
    n_features = 20
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create imbalanced target (70% class 0, 30% class 1)
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Create group column (e.g., user_id or region)
    group_ids = np.random.choice(range(50), size=n_samples)
    
    # Create categorical stratification column
    categories = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Create DataFrames
    positive_df = pd.DataFrame(X[y == 1], 
                              columns=[f'feature_{i}' for i in range(n_features)])
    positive_df['target'] = 1
    positive_df['group_id'] = group_ids[y == 1]
    positive_df['category'] = categories[y == 1]
    
    negative_df = pd.DataFrame(X[y == 0], 
                              columns=[f'feature_{i}' for i in range(n_features)])
    negative_df['target'] = 0
    negative_df['group_id'] = group_ids[y == 0]
    negative_df['category'] = categories[y == 0]
    
    return positive_df, negative_df


def test_basic_stratification():
    """Test basic stratification by target column"""
    print("Testing basic stratification...")
    
    positive_df, negative_df = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        pos_file = os.path.join(temp_dir, 'positive.parquet')
        neg_file = os.path.join(temp_dir, 'negative.parquet')
        positive_df.to_parquet(pos_file)
        negative_df.to_parquet(neg_file)
        
        # Prepare datasets with basic stratification
        output_dir = os.path.join(temp_dir, 'output')
        train_df, val_df, test_df = prepare_datasets(
            positive_file=pos_file,
            negative_file=neg_file,
            output_dir=output_dir,
            target_column='target',
            stratify_by='target'
        )
        
        # Verify stratification maintained proportions
        orig_pos_ratio = len(positive_df) / (len(positive_df) + len(negative_df))
        train_pos_ratio = len(train_df[train_df.target == 1]) / len(train_df)
        val_pos_ratio = len(val_df[val_df.target == 1]) / len(val_df)
        test_pos_ratio = len(test_df[test_df.target == 1]) / len(test_df)
        
        print(f"Original positive ratio: {orig_pos_ratio:.3f}")
        print(f"Train positive ratio: {train_pos_ratio:.3f}")
        print(f"Val positive ratio: {val_pos_ratio:.3f}")
        print(f"Test positive ratio: {test_pos_ratio:.3f}")
        
        # Check that ratios are similar (within 5%)
        assert abs(train_pos_ratio - orig_pos_ratio) < 0.05
        assert abs(val_pos_ratio - orig_pos_ratio) < 0.05
        assert abs(test_pos_ratio - orig_pos_ratio) < 0.05
        
        print("✓ Basic stratification test passed\n")


def test_categorical_stratification():
    """Test stratification by categorical column"""
    print("Testing categorical stratification...")
    
    positive_df, negative_df = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        pos_file = os.path.join(temp_dir, 'positive.parquet')
        neg_file = os.path.join(temp_dir, 'negative.parquet')
        positive_df.to_parquet(pos_file)
        negative_df.to_parquet(neg_file)
        
        # Prepare datasets with categorical stratification
        output_dir = os.path.join(temp_dir, 'output')
        train_df, val_df, test_df = prepare_datasets(
            positive_file=pos_file,
            negative_file=neg_file,
            output_dir=output_dir,
            target_column='target',
            stratify_by='category'
        )
        
        # Verify categorical proportions maintained
        full_df = pd.concat([positive_df, negative_df])
        orig_cat_dist = full_df['category'].value_counts(normalize=True).sort_index()
        train_cat_dist = train_df['category'].value_counts(normalize=True).sort_index()
        val_cat_dist = val_df['category'].value_counts(normalize=True).sort_index()
        test_cat_dist = test_df['category'].value_counts(normalize=True).sort_index()
        
        print("Original category distribution:")
        print(orig_cat_dist)
        print("\nTrain category distribution:")
        print(train_cat_dist)
        
        # Check that distributions are similar
        for cat in orig_cat_dist.index:
            assert abs(train_cat_dist[cat] - orig_cat_dist[cat]) < 0.1
            assert abs(val_cat_dist[cat] - orig_cat_dist[cat]) < 0.1
            assert abs(test_cat_dist[cat] - orig_cat_dist[cat]) < 0.1
        
        print("✓ Categorical stratification test passed\n")


def test_group_aware_splits():
    """Test group-aware splitting"""
    print("Testing group-aware splits...")
    
    positive_df, negative_df = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        pos_file = os.path.join(temp_dir, 'positive.parquet')
        neg_file = os.path.join(temp_dir, 'negative.parquet')
        positive_df.to_parquet(pos_file)
        negative_df.to_parquet(neg_file)
        
        # Prepare datasets with group-aware splitting
        output_dir = os.path.join(temp_dir, 'output')
        train_df, val_df, test_df = prepare_datasets(
            positive_file=pos_file,
            negative_file=neg_file,
            output_dir=output_dir,
            target_column='target',
            stratify_groups='group_id',
            stratify_groups_min_size=5
        )
        
        # Verify no group overlap between splits
        train_groups = set(train_df['group_id'].unique())
        val_groups = set(val_df['group_id'].unique())
        test_groups = set(test_df['group_id'].unique())
        
        print(f"Train groups: {len(train_groups)}")
        print(f"Val groups: {len(val_groups)}")
        print(f"Test groups: {len(test_groups)}")
        
        # Check no overlap
        assert len(train_groups & val_groups) == 0
        assert len(train_groups & test_groups) == 0
        assert len(val_groups & test_groups) == 0
        
        # Check minimum group sizes
        train_group_sizes = train_df['group_id'].value_counts()
        val_group_sizes = val_df['group_id'].value_counts()
        test_group_sizes = test_df['group_id'].value_counts()
        
        assert train_group_sizes.min() >= 5
        assert val_group_sizes.min() >= 5
        assert test_group_sizes.min() >= 5
        
        print("✓ Group-aware splits test passed\n")


def test_class_balancing():
    """Test class balancing functionality"""
    print("Testing class balancing...")
    
    positive_df, negative_df = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        pos_file = os.path.join(temp_dir, 'positive.parquet')
        neg_file = os.path.join(temp_dir, 'negative.parquet')
        positive_df.to_parquet(pos_file)
        negative_df.to_parquet(neg_file)
        
        # Prepare datasets with class balancing
        output_dir = os.path.join(temp_dir, 'output')
        train_df, val_df, test_df = prepare_datasets(
            positive_file=pos_file,
            negative_file=neg_file,
            output_dir=output_dir,
            target_column='target',
            balance_classes=True,
            max_imbalance_ratio=2.0
        )
        
        # Check class balance in training set
        train_class_counts = train_df['target'].value_counts()
        imbalance_ratio = train_class_counts.max() / train_class_counts.min()
        
        print(f"Original imbalance ratio: {len(negative_df) / len(positive_df):.3f}")
        print(f"Balanced imbalance ratio: {imbalance_ratio:.3f}")
        print(f"Train class distribution:\n{train_class_counts}")
        
        # Check that imbalance ratio is within limits
        assert imbalance_ratio <= 2.0
        
        print("✓ Class balancing test passed\n")


def test_combined_features():
    """Test combination of multiple stratification features"""
    print("Testing combined stratification features...")
    
    positive_df, negative_df = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        pos_file = os.path.join(temp_dir, 'positive.parquet')
        neg_file = os.path.join(temp_dir, 'negative.parquet')
        positive_df.to_parquet(pos_file)
        negative_df.to_parquet(neg_file)
        
        # Prepare datasets with multiple features
        output_dir = os.path.join(temp_dir, 'output')
        train_df, val_df, test_df = prepare_datasets(
            positive_file=pos_file,
            negative_file=neg_file,
            output_dir=output_dir,
            target_column='target',
            stratify_by='target',
            balance_classes=True,
            max_imbalance_ratio=1.5
        )
        
        # Verify both stratification and balancing worked
        train_class_counts = train_df['target'].value_counts()
        imbalance_ratio = train_class_counts.max() / train_class_counts.min()
        
        print(f"Combined features - imbalance ratio: {imbalance_ratio:.3f}")
        print(f"Train class distribution:\n{train_class_counts}")
        
        assert imbalance_ratio <= 1.5
        
        print("✓ Combined features test passed\n")


def main():
    """Run all stratification tests"""
    print("Running comprehensive stratification tests...\n")
    
    try:
        test_basic_stratification()
        test_categorical_stratification()
        test_group_aware_splits()
        test_class_balancing()
        test_combined_features()
        
        print("🎉 All stratification tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
