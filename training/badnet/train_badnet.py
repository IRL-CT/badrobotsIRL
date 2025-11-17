#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BADNet PyTorch - Standalone Training Script

Run training without W&B sweep for testing or single configuration runs.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import wandb

from badnet_pytorch import (
    set_seed, BadNetDataset, BadNetCNN, BadNetDatasetWithAugmentation,
    BadNetPretrained, BadNetSimple, create_model,  # Add these imports
    create_interparticipant_folds, train_fold
)
from get_metrics import get_test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train BADNet PyTorch model")
    
    # Data paths
    parser.add_argument("--csv_path", type=str, 
                        default="../../preprocessing/full_features/all_participants_0_3.csv",
                        help="Path to CSV file with labels")
    parser.add_argument("--image_base_path", type=str, 
                        default="../../../data/frames",
                        help="Base path to image folders")
    
    # Model hyperparameters
    parser.add_argument("--activation", type=str, default="relu", 
                        choices=["relu", "sigmoid"], help="Activation function")
    parser.add_argument("--kernel_size", type=int, default=4, help="Kernel size for conv layers")
    parser.add_argument("--base_filters", type=int, default=16, help="Base number of filters")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    # Cross-validation
    parser.add_argument("--num_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--fold", type=int, default=None, 
                        help="Specific fold to train (None for all folds)")
    
    # Label type
    parser.add_argument("--label_type", type=str, default="binary", 
                        choices=["binary", "multiclass"], help="Label type")
    
    # Participants
    parser.add_argument("--exclude_participants", nargs="*", default=[], 
                        help="Participants to exclude")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--wandb_project", type=str, default="brirl_BADNet", 
                        help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    return parser.parse_args()


def evaluate_model(model, dataloader, device, num_classes=2):
    """
    Evaluate model and return metrics using get_test_metrics.
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Get metrics using get_test_metrics (with tolerance)
    metrics = get_test_metrics(all_preds, all_labels, tolerance=1)
    
    # Add kappa and confusion matrix
    kappa = cohen_kappa_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics['kappa'] = kappa
    metrics['confusion_matrix'] = conf_matrix
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1 Score:  {metrics['test_f1']:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"\nTolerant Metrics (tolerance=1):")
    print(f"Accuracy:  {metrics['test_accuracy_tolerant']:.4f}")
    print(f"Precision: {metrics['test_precision_tolerant']:.4f}")
    print(f"Recall:    {metrics['test_recall_tolerant']:.4f}")
    print(f"F1 Score:  {metrics['test_f1_tolerant']:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Sweep configuration
    sweep_config = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "avg_test_f1"},
        "parameters": {
            "activation": {"values": ['relu', 'sigmoid']},
            "kernel_size": {"values": [2, 4, 6, 8]},
            "base_filters": {"values": [16, 32, 64]},
            "learning_rate": {"values": [0.0001, 0.001, 0.01]},
            "batch_size": {"values": [16, 32, 64]},
            "seed": {"values": [42, 1369]},
            "epochs": {"values": [100]},
            "label_type": {"values": ['binary', 'multiclass']},
            "num_folds": {"values": [5]},
            "csv_path": {"values": [args.csv_path]},
            "image_base_path": {"values": [args.image_base_path]},
            "exclude_participants": {"values": [args.exclude_participants]},
            "patience": {"values": [args.patience]},
            "num_workers": {"values": [args.num_workers]},
            "num_augmentations": {"values": [0,2,3]},
            "model_type": {"values": ['original', 'simple', 'pretrained_resnet18', 'pretrained_resnet34']},
            "freeze_backbone": {"values": [True, False]},
            "dropout": {"values": [0.3, 0.5, 0.7]},
        },
    }
    
    def train_wrapper():
        wandb.init()
        config = wandb.config
        
        # Override args with sweep config
        args.activation = config.activation
        args.kernel_size = config.kernel_size
        args.base_filters = config.base_filters
        args.learning_rate = config.learning_rate
        args.batch_size = config.batch_size
        args.seed = config.seed
        args.epochs = config.epochs
        args.label_type = config.label_type
        args.num_folds = config.num_folds
        args.csv_path = config.csv_path
        args.image_base_path = config.image_base_path
        args.exclude_participants = config.exclude_participants
        args.patience = config.patience
        args.num_workers = config.num_workers
        args.num_augmentations = config.num_augmentations
        args.model_type = config.model_type
        args.freeze_backbone = config.freeze_backbone
        args.dropout = config.dropout
        
        set_seed(args.seed)
        
        # Determine number of classes
        num_classes = 2 if args.label_type == 'binary' else 4
        print(f"Label type: {args.label_type}, Num classes: {num_classes}")
        
        # Load data
        print(f"Loading data from {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        print(f"Loaded {len(df)} rows")
        print(f"Unique participants: {df['participant'].nunique()}")
        
        # Create folds
        print(f"\nCreating {args.num_folds} inter-participant folds...")
        folds = create_interparticipant_folds(
            df, 
            num_folds=args.num_folds,
            exclude_participants=args.exclude_participants,
            seed=args.seed
        )
        
        # Determine which folds to train
        if args.fold is not None:
            fold_indices = [args.fold]
        else:
            fold_indices = list(range(len(folds)))
        
        all_test_metrics = []
        
        for fold_idx in fold_indices:
            train_participants, val_participants, test_participants = folds[fold_idx]
            
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx}")
            print(f"{'='*60}")
            print(f"Train participants: {train_participants}")
            print(f"Val participants: {val_participants}")
            print(f"Test participants: {test_participants}")
            
            # Create datasets
            print("\nLoading datasets...")
            train_dataset = BadNetDatasetWithAugmentation(df, train_participants, args.image_base_path,
                                                           label_type=args.label_type, num_augmentations=args.num_augmentations)
            val_dataset = BadNetDatasetWithAugmentation(df, val_participants, args.image_base_path,
                                                         label_type=args.label_type, num_augmentations=0)
            test_dataset = BadNetDatasetWithAugmentation(df, test_participants, args.image_base_path,
                                                          label_type=args.label_type, num_augmentations=0)
            
            if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
                print(f"Warning: Empty dataset in fold {fold_idx}. Skipping...")
                continue
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=use_cuda)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=use_cuda)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=use_cuda)
            
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            
            # Create model
            print("\nCreating model...")
            model = model = create_model(
                model_type=args.model_type,
                num_classes=num_classes,
                base_filters=args.base_filters,
                kernel_size=args.kernel_size,
                activation=args.activation,
                freeze_backbone=args.freeze_backbone,
                dropout=args.dropout
            )
            model = model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # Train
            checkpoint_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}')
            print(f"\nStarting training for {args.epochs} epochs...")
            print(f"Checkpoints will be saved to {checkpoint_dir}")
            
            history = train_fold(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, args.epochs, patience=args.patience, checkpoint_dir=checkpoint_dir
            )
            
            # Log training history to W&B
            for epoch_idx in range(len(history['train_loss'])):
                wandb.log({
                    f'fold_{fold_idx}_train_loss': history['train_loss'][epoch_idx],
                    f'fold_{fold_idx}_train_acc': history['train_acc'][epoch_idx],
                    f'fold_{fold_idx}_val_loss': history['val_loss'][epoch_idx],
                    f'fold_{fold_idx}_val_acc': history['val_acc'][epoch_idx],
                    'epoch': epoch_idx + 1,
                    'fold': fold_idx
                })
            
            # Evaluate
            print(f"\nEvaluating on test set...")
            metrics = evaluate_model(model, test_loader, device, num_classes=num_classes)
            all_test_metrics.append(metrics)
            
            # Log test metrics to W&B
            wandb.log({
                f'fold_{fold_idx}_test_accuracy': metrics['test_accuracy'],
                f'fold_{fold_idx}_test_precision': metrics['test_precision'],
                f'fold_{fold_idx}_test_recall': metrics['test_recall'],
                f'fold_{fold_idx}_test_f1': metrics['test_f1'],
                f'fold_{fold_idx}_test_accuracy_tolerant': metrics['test_accuracy_tolerant'],
                f'fold_{fold_idx}_test_precision_tolerant': metrics['test_precision_tolerant'],
                f'fold_{fold_idx}_test_recall_tolerant': metrics['test_recall_tolerant'],
                f'fold_{fold_idx}_test_f1_tolerant': metrics['test_f1_tolerant'],
                f'fold_{fold_idx}_test_kappa': metrics['kappa']
            })
            
            # Log prediction probabilities
            probs_df = pd.DataFrame(metrics['probabilities'])
            table = wandb.Table(dataframe=probs_df)
            wandb.log({f"fold_{fold_idx}_prediction_probabilities_table": table})
            
            # Save final model
            final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fold': fold_idx,
                'metrics': metrics,
                'args': vars(args)
            }, final_model_path)
            print(f"Saved final model to {final_model_path}")
        
        # Summary
        if len(all_test_metrics) > 0:
            print(f"\n{'='*60}")
            print("SUMMARY ACROSS ALL FOLDS")
            print(f"{'='*60}")
            
            avg_accuracy = np.mean([m['test_accuracy'] for m in all_test_metrics])
            avg_precision = np.mean([m['test_precision'] for m in all_test_metrics])
            avg_recall = np.mean([m['test_recall'] for m in all_test_metrics])
            avg_f1 = np.mean([m['test_f1'] for m in all_test_metrics])
            avg_kappa = np.mean([m['kappa'] for m in all_test_metrics])
            
            avg_accuracy_tol = np.mean([m['test_accuracy_tolerant'] for m in all_test_metrics])
            avg_precision_tol = np.mean([m['test_precision_tolerant'] for m in all_test_metrics])
            avg_recall_tol = np.mean([m['test_recall_tolerant'] for m in all_test_metrics])
            avg_f1_tol = np.mean([m['test_f1_tolerant'] for m in all_test_metrics])
            
            std_accuracy = np.std([m['test_accuracy'] for m in all_test_metrics])
            std_f1 = np.std([m['test_f1'] for m in all_test_metrics])
            std_accuracy_tol = np.std([m['test_accuracy_tolerant'] for m in all_test_metrics])
            std_f1_tol = np.std([m['test_f1_tolerant'] for m in all_test_metrics])
            
            print(f"Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"Precision: {avg_precision:.4f}")
            print(f"Recall:    {avg_recall:.4f}")
            print(f"F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
            print(f"Kappa:     {avg_kappa:.4f}")
            print(f"\nTolerant Metrics:")
            print(f"Accuracy (tol):  {avg_accuracy_tol:.4f} ± {std_accuracy_tol:.4f}")
            print(f"Precision (tol): {avg_precision_tol:.4f}")
            print(f"Recall (tol):    {avg_recall_tol:.4f}")
            print(f"F1 Score (tol):  {avg_f1_tol:.4f} ± {std_f1_tol:.4f}")
            
            # Log average metrics to W&B
            wandb.log({
                'avg_test_accuracy': avg_accuracy,
                'avg_test_precision': avg_precision,
                'avg_test_recall': avg_recall,
                'avg_test_f1': avg_f1,
                'avg_test_kappa': avg_kappa,
                'avg_test_accuracy_tolerant': avg_accuracy_tol,
                'avg_test_precision_tolerant': avg_precision_tol,
                'avg_test_recall_tolerant': avg_recall_tol,
                'avg_test_f1_tolerant': avg_f1_tol,
                'std_test_accuracy': std_accuracy,
                'std_test_f1': std_f1,
                'std_test_accuracy_tolerant': std_accuracy_tol,
                'std_test_f1_tolerant': std_f1_tol
            })
            
            wandb.run.summary.update({
                'avg_test_accuracy': avg_accuracy,
                'avg_test_precision': avg_precision,
                'avg_test_recall': avg_recall,
                'avg_test_f1': avg_f1,
                'avg_test_kappa': avg_kappa,
                'avg_test_accuracy_tolerant': avg_accuracy_tol,
                'avg_test_precision_tolerant': avg_precision_tol,
                'avg_test_recall_tolerant': avg_recall_tol,
                'avg_test_f1_tolerant': avg_f1_tol
            })
        
        wandb.finish()
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    print(f"Sweep ID: {sweep_id}")
    print("Starting sweep agent...")
    
    # Run sweep
    wandb.agent(sweep_id, function=train_wrapper)


if __name__ == "__main__":
    main()