#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inter-participant data splits for image classification.

This module provides utilities for creating train/val/test splits
based on participant grouping (inter-participant cross-validation).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def create_interparticipant_folds(df, num_folds=5, exclude_participants=None, seed=42):
    """
    Create inter-participant folds for cross-validation.
    
    Args:
        df: DataFrame with 'participant' column
        num_folds: Number of folds (default 5)
        exclude_participants: List of participant IDs to exclude (default None)
        seed: Random seed for reproducibility
    
    Returns:
        List of tuples: [(train_participants, val_participants, test_participants), ...]
    """
    np.random.seed(seed)
    
    # Get unique participants
    all_participants = df['participant'].unique().tolist()
    
    # Exclude specified participants
    if exclude_participants:
        all_participants = [p for p in all_participants if p not in exclude_participants]
        print(f"Excluded {len(exclude_participants)} participants. Remaining: {len(all_participants)}")
    
    # Shuffle participants
    np.random.shuffle(all_participants)
    
    # Split participants into num_folds groups
    fold_size = len(all_participants) // num_folds
    participant_groups = []
    
    for i in range(num_folds):
        if i == num_folds - 1:
            # Last fold gets remaining participants
            participant_groups.append(all_participants[i * fold_size:])
        else:
            participant_groups.append(all_participants[i * fold_size:(i + 1) * fold_size])
    
    # Create folds: each group takes turn being test set
    folds = []
    for fold_idx in range(num_folds):
        test_participants = participant_groups[fold_idx]
        
        # Remaining participants for train and val
        train_val_participants = []
        for i in range(num_folds):
            if i != fold_idx:
                train_val_participants.extend(participant_groups[i])
        
        np.random.shuffle(train_val_participants)
        
        # Split into train (75% of remaining = 60% total) and val (25% of remaining = 20% total)
        split_idx = int(len(train_val_participants) * 0.75)
        train_participants = train_val_participants[:split_idx]
        val_participants = train_val_participants[split_idx:]
        
        folds.append((train_participants, val_participants, test_participants))
        
        print(f"Fold {fold_idx}: Train={len(train_participants)} participants, "
              f"Val={len(val_participants)} participants, "
              f"Test={len(test_participants)} participants")
    
    return folds


def create_interparticipant_folds_custom_ratio(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                                                num_folds=5, exclude_participants=None, seed=42):
    """
    Create inter-participant folds with custom train/val/test ratios.
    
    Note: For K-fold CV, the test ratio is approximately 1/num_folds.
    The train_ratio and val_ratio are applied to the remaining participants.
    
    Args:
        df: DataFrame with 'participant' column
        train_ratio: Proportion for training (of train+val)
        val_ratio: Proportion for validation (of train+val)
        test_ratio: Proportion for testing (automatically set to 1/num_folds)
        num_folds: Number of folds
        exclude_participants: List of participant IDs to exclude
        seed: Random seed
    
    Returns:
        List of tuples: [(train_participants, val_participants, test_participants), ...]
    """
    np.random.seed(seed)
    
    # Get unique participants
    all_participants = df['participant'].unique().tolist()
    
    # Exclude specified participants
    if exclude_participants:
        all_participants = [p for p in all_participants if p not in exclude_participants]
    
    np.random.shuffle(all_participants)
    
    # Create folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    folds = []
    participant_array = np.array(all_participants)
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(participant_array)):
        test_participants = participant_array[test_idx].tolist()
        train_val_participants = participant_array[train_val_idx].tolist()
        
        np.random.shuffle(train_val_participants)
        
        # Normalize train/val ratio
        total_ratio = train_ratio + val_ratio
        normalized_train_ratio = train_ratio / total_ratio
        
        split_idx = int(len(train_val_participants) * normalized_train_ratio)
        train_participants = train_val_participants[:split_idx]
        val_participants = train_val_participants[split_idx:]
        
        folds.append((train_participants, val_participants, test_participants))
    
    return folds


def get_participant_statistics(df, participants):
    """
    Get statistics for a set of participants.
    
    Args:
        df: DataFrame with participant, binary_label, multiclass_label columns
        participants: List of participant IDs
    
    Returns:
        Dictionary with statistics
    """
    subset = df[df['participant'].isin(participants)]
    
    binary_dist = subset['binary_label'].value_counts().to_dict()
    multiclass_dist = subset['multiclass_label'].value_counts().to_dict()
    
    return {
        'num_participants': len(participants),
        'total_samples': len(subset),
        'samples_per_participant': len(subset) / len(participants) if len(participants) > 0 else 0,
        'binary_label_distribution': binary_dist,
        'multiclass_label_distribution': multiclass_dist
    }


def print_fold_statistics(df, folds):
    """
    Print detailed statistics for each fold.
    
    Args:
        df: DataFrame with data
        folds: List of (train_participants, val_participants, test_participants)
    """
    for fold_idx, (train_p, val_p, test_p) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx} STATISTICS")
        print(f"{'='*50}")
        
        train_stats = get_participant_statistics(df, train_p)
        val_stats = get_participant_statistics(df, val_p)
        test_stats = get_participant_statistics(df, test_p)
        
        print(f"\nTRAIN SET:")
        print(f"  Participants: {train_p}")
        print(f"  Total samples: {train_stats['total_samples']}")
        print(f"  Binary label distribution: {train_stats['binary_label_distribution']}")
        print(f"  Multiclass label distribution: {train_stats['multiclass_label_distribution']}")
        
        print(f"\nVALIDATION SET:")
        print(f"  Participants: {val_p}")
        print(f"  Total samples: {val_stats['total_samples']}")
        print(f"  Binary label distribution: {val_stats['binary_label_distribution']}")
        print(f"  Multiclass label distribution: {val_stats['multiclass_label_distribution']}")
        
        print(f"\nTEST SET:")
        print(f"  Participants: {test_p}")
        print(f"  Total samples: {test_stats['total_samples']}")
        print(f"  Binary label distribution: {test_stats['binary_label_distribution']}")
        print(f"  Multiclass label distribution: {test_stats['multiclass_label_distribution']}")


def validate_image_paths(df, image_base_path):
    """
    Validate that image files exist for all frames in the dataframe.
    
    Args:
        df: DataFrame with frame and participant columns
        image_base_path: Base path to image folders
    
    Returns:
        Tuple of (valid_count, missing_count, missing_paths)
    """
    import os
    
    valid_count = 0
    missing_count = 0
    missing_paths = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        participant = row['participant']
        frame = int(row['frame'])
        image_path = os.path.join(image_base_path, participant, f"{frame}.jpg")
        
        if os.path.exists(image_path):
            valid_count += 1
        else:
            missing_count += 1
            if len(missing_paths) < 10:  # Only store first 10 missing paths
                missing_paths.append(image_path)
    
    return valid_count, missing_count, missing_paths


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create inter-participant folds")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exclude", nargs="+", default=[], help="Participants to exclude")
    parser.add_argument("--image_path", type=str, default=None, help="Path to validate images")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")
    print(f"Unique participants: {df['participant'].unique().tolist()}")
    
    # Create folds
    folds = create_interparticipant_folds(df, num_folds=args.num_folds, 
                                           exclude_participants=args.exclude, seed=args.seed)
    
    # Print statistics
    print_fold_statistics(df, folds)
    
    # Validate images if path provided
    if args.image_path:
        print(f"\nValidating image paths at {args.image_path}...")
        valid, missing, missing_paths = validate_image_paths(df, args.image_path)
        print(f"Valid images: {valid}")
        print(f"Missing images: {missing}")
        if missing_paths:
            print(f"Sample missing paths: {missing_paths}")
