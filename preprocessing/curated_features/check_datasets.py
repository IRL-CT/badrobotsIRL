#!/usr/bin/env python3
"""
Script to check for NaN values in datasets and replace them with 0.
"""

import pandas as pd
import numpy as np
import os

def check_and_fix_nan(file_path, dataset_name):
    """
    Check for NaN values in a dataset and replace them with 0.
    
    Args:
        file_path (str): Path to the CSV file
        dataset_name (str): Name of the dataset for logging
    """
    print(f"\n{'='*60}")
    print(f"Checking dataset: {dataset_name}")
    print(f"File: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"📊 Dataset shape: {df.shape}")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans == 0:
            print("✅ No NaN values found!")
            return
        
        print(f"⚠️  Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
        
        # Show columns with NaN values
        nan_columns = nan_counts[nan_counts > 0]
        print("\n📋 Columns with NaN values:")
        for col, count in nan_columns.items():
            percentage = (count / len(df)) * 100
            print(f"   {col}: {count} NaNs ({percentage:.2f}%)")
        
        # Show sample of rows with NaN values
        nan_rows = df[df.isnull().any(axis=1)]
        if len(nan_rows) > 0:
            print(f"\n🔍 Sample of rows with NaN values (showing first 5):")
            print(nan_rows.head())
        
        # Replace NaN values with 0
        df_cleaned = df.fillna(0)
        
        # Verify no NaN values remain
        remaining_nans = df_cleaned.isnull().sum().sum()
        if remaining_nans == 0:
            print("✅ Successfully replaced all NaN values with 0")
        else:
            print(f"❌ Still {remaining_nans} NaN values remaining!")
            return
        
        # Save the cleaned dataset
        backup_path = file_path.replace('.csv', '_backup.csv')
        
        # Create backup of original
        df.to_csv(backup_path, index=False)
        print(f"💾 Backup saved to: {backup_path}")
        
        # Save cleaned dataset
        df_cleaned.to_csv(file_path, index=False)
        print(f"💾 Cleaned dataset saved to: {file_path}")
        
        # Show summary statistics
        print(f"\n📈 Summary:")
        print(f"   - Original NaN values: {total_nans}")
        print(f"   - Values replaced with 0: {total_nans}")
        print(f"   - Affected columns: {len(nan_columns)}")
        print(f"   - Total rows: {len(df)}")
        print(f"   - Total columns: {len(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {str(e)}")

def main():
    """Main function to check all datasets."""
    
    print("🔍 NaN Value Checker and Fixer")
    print("=" * 60)
    
    # Define datasets to check
    datasets = [
        {
            'name': 'Curated Features Dataset v3',
            'path': 'curated_features_dataset_v3.csv'
        },
        {
            'name': 'Curated Features Dataset V1',
            'path': 'curated_features_dataset_v1.csv'
        },
        {
            'name': 'Full Features',
            'path': '../full_features/all_participants_0_3.csv'
        },
        {
            'name': 'Stats Features',
            'path': '../stats_features/all_participants_stats_0_3.csv'
        },
        {
            'name': 'RF Features',
            'path': '../rf_features/all_participants_rf_0_3_40.csv'
        },
        {
            'name': 'Text Embeddings',
            'path': '../clip_text_embeddings.csv'
        },
        {
            'name': 'Text Embeddings PCA',
            'path': '../clip_text_embeddings_pca.csv'
        },
        { 'name': 'cosine similarity',
          'path': '../clip_text_cosine_similarity.csv'
        }
    ]
    
    # Check each dataset
    for dataset in datasets:
        check_and_fix_nan(dataset['path'], dataset['name'])
    
    print(f"\n{'='*60}")
    print("🎉 Finished checking all datasets!")
    print("=" * 60)

if __name__ == "__main__":
    main()
