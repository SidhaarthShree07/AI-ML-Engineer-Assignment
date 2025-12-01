"""
Create a smaller test dataset by sampling half the data from the original dataset.

This script:
1. Reads train.csv and test.csv from data/ folder
2. Samples 50% of rows (or configurable fraction)
3. Saves to data_test/ folder for quick functionality testing

Usage:
    python create_test_dataset.py                    # Default: 50% sample
    python create_test_dataset.py --fraction 0.25   # 25% sample
    python create_test_dataset.py --fraction 0.1    # 10% sample (very fast testing)
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def create_test_dataset(
    source_dir: str = "data_test",
    target_dir: str = "data_test", 
    fraction: float = 0.9,
    random_state: int = 42
):
    """
    Create a smaller test dataset by sampling from the original.
    
    Args:
        source_dir: Source directory containing train/ and test/ subdirectories
        target_dir: Target directory for the smaller dataset
        fraction: Fraction of data to keep (0.5 = 50%)
        random_state: Random seed for reproducibility
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print(f"Creating test dataset with {fraction*100:.0f}% of original data")
    print(f"Source: {source_path.absolute()}")
    print(f"Target: {target_path.absolute()}")
    print("-" * 60)
    
    # Create target directories
    (target_path / "train").mkdir(parents=True, exist_ok=True)
    (target_path / "test").mkdir(parents=True, exist_ok=True)
    
    # Process train data
    train_source = source_path / "train" / "train.csv"
    train_target = target_path / "train" / "train.csv"
    
    if train_source.exists():
        print(f"Reading: {train_source}")
        train_df = pd.read_csv(train_source)
        original_train_rows = len(train_df)
        
        # Sample the data
        train_sample = train_df.sample(frac=fraction, random_state=random_state)
        train_sample = train_sample.reset_index(drop=True)
        
        # Save sampled data
        train_sample.to_csv(train_target, index=False)
        print(f"  Train: {original_train_rows:,} -> {len(train_sample):,} rows ({fraction*100:.0f}%)")
    else:
        print(f"Warning: {train_source} not found")
    
    # Process test data
    test_source = source_path / "test" / "test.csv"
    test_target = target_path / "test" / "test.csv"
    
    if test_source.exists():
        print(f"Reading: {test_source}")
        test_df = pd.read_csv(test_source)
        original_test_rows = len(test_df)
        
        # Sample the data
        test_sample = test_df.sample(frac=fraction, random_state=random_state)
        test_sample = test_sample.reset_index(drop=True)
        
        # Save sampled data
        test_sample.to_csv(test_target, index=False)
        print(f"  Test:  {original_test_rows:,} -> {len(test_sample):,} rows ({fraction*100:.0f}%)")
    else:
        print(f"Warning: {test_source} not found")
    
    # Copy sample_submission.csv if exists
    sample_sub_source = source_path / "sample_submission.csv"
    sample_sub_target = target_path / "sample_submission.csv"
    
    if sample_sub_source.exists():
        print(f"Copying: sample_submission.csv")
        sample_sub_df = pd.read_csv(sample_sub_source)
        
        # If test data was sampled, filter sample_submission to match
        if test_source.exists() and 'id' in sample_sub_df.columns:
            test_ids = set(test_sample['id'].values) if 'id' in test_sample.columns else None
            if test_ids:
                sample_sub_filtered = sample_sub_df[sample_sub_df['id'].isin(test_ids)]
                sample_sub_filtered.to_csv(sample_sub_target, index=False)
                print(f"  Sample submission: {len(sample_sub_df):,} -> {len(sample_sub_filtered):,} rows")
            else:
                sample_sub_df.to_csv(sample_sub_target, index=False)
        else:
            sample_sub_df.to_csv(sample_sub_target, index=False)
    
    print("-" * 60)
    print(f"Test dataset created successfully at: {target_path.absolute()}")
    print()
    print("To run the agent with this test dataset:")
    print(f"  python hybrid_agent.py --dataset_path {target_path} --competition_id test --output_dir ./output")
    

def main():
    parser = argparse.ArgumentParser(
        description="Create a smaller test dataset for quick functionality testing"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data",
        help="Source directory containing train/ and test/ (default: data)"
    )
    parser.add_argument(
        "--target_dir", 
        type=str,
        default="data_test",
        help="Target directory for smaller dataset (default: data_test)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.005,
        help="Fraction of data to keep (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_test_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        fraction=args.fraction,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
