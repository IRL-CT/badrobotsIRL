#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BADNet Robot Behavior Classification - PyTorch Implementation

CNN model to classify robot behavior based on human bystander reactions.
Supports inter-participant cross-validation and W&B sweeps.
"""

import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)
from sklearn.model_selection import KFold
import wandb


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BadNetDataset(Dataset):
    """
    Dataset for loading robot behavior images with frame-to-label mapping.
    
    Args:
        df: DataFrame with columns [frame, participant, binary_label, multiclass_label, ...]
        participants: List of participant IDs to include
        image_base_path: Base path to image folders
        label_type: 'binary' or 'multiclass'
        transform: Image transformations
    """
    def __init__(self, df, participants, image_base_path, label_type='binary', transform=None, cache_images=True):
        self.df = df[df['participant'].isin(participants)].reset_index(drop=True)
        self.image_base_path = image_base_path
        self.label_type = label_type
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        # Build list of (image_path, label) tuples
        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            participant = row['participant']
            frame = int(row['frame'])
            
            if label_type == 'binary':
                label = int(row['binary_label'])
            else:  # multiclass
                label = int(row['multiclass_label'])
            
            image_path = os.path.join(image_base_path, participant, f"{frame}.jpg")
            if os.path.exists(image_path):
                self.samples.append((image_path, label))
            else:
                print(f"Warning: Image not found: {image_path}")

        if cache_images:
            print("Caching images in memory...")
            for i, (image_path, label) in enumerate(self.samples):
                image = Image.open(image_path).convert('RGB')
                if transform:
                    image = transform(image)
                self.image_cache[image_path] = image
                if i % 1000 == 0:
                    print(f"Cached {i}/{len(self.samples)} images")

        
        print(f"Loaded {len(self.samples)} samples from {len(participants)} participants")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        if self.cache_images and image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        return image, label


class BadNetCNN(nn.Module):
    """
    CNN model for robot behavior classification matching the original BadNet architecture.
    
    Args:
        num_classes: Number of output classes (2 for binary, 4 for multiclass)
        base_filters: Base number of filters in first conv layer
        kernel_size: Kernel size for conv layers
        activation: Activation function ('relu' or 'sigmoid')
    """
    def __init__(self, num_classes=2, base_filters=16, kernel_size=4, activation='relu'):
        super(BadNetCNN, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(3, base_filters, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.dropout1 = nn.Dropout(0.15)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.dropout3 = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm2d(base_filters*4)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.dropout4 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(base_filters*4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights with He uniform
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.batchnorm(x)
        
        # Global Average Pooling and FC
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout4(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x


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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, device, num_classes=2):
    """
    Evaluate model and return metrics using get_test_metrics.
    
    Returns:
        Dictionary of evaluation metrics
    """
    from get_metrics import get_test_metrics
    
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


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, epochs, patience=20, checkpoint_dir='./checkpoints'):
    """
    Train model for one fold with early stopping.
    
    Returns:
        Training history dictionary
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(checkpoint_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def main_wrapper():
    """Wrapper function for W&B sweep."""
    
    def main_func():
        # Initialize W&B run
        wandb.init()
        config = wandb.config
        
        # Set seed
        seed = config.get('seed', 42)
        set_seed(seed)
        
        # Extract hyperparameters
        activation = config.get('activation', 'relu')
        kernel_size = config.get('kernel_size', 4)
        base_filters = config.get('base_filters', 16)
        learning_rate = config.get('learning_rate', 0.0001)
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 100)
        label_type = config.get('label_type', 'binary')
        num_folds = config.get('num_folds', 5)
        
        # Paths
        csv_path = config.get('csv_path', '../../preprocessing/full_features/all_participants_0_3.csv')
        image_base_path = config.get('image_base_path', '../../data/frames')
        exclude_participants = config.get('exclude_participants', [])
        
        # Determine number of classes
        num_classes = 2 if label_type == 'binary' else 4
        
        print(f"Configuration:")
        print(f"  Label Type: {label_type}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Activation: {activation}")
        print(f"  Kernel Size: {kernel_size}")
        print(f"  Base Filters: {base_filters}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Excluded Participants: {exclude_participants}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Create image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create inter-participant folds
        folds = create_interparticipant_folds(df, num_folds=num_folds, 
                                               exclude_participants=exclude_participants, seed=seed)
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        all_test_metrics = []
        
        for fold_idx, (train_participants, val_participants, test_participants) in enumerate(folds):
            print(f"\n{'='*50}")
            print(f"FOLD {fold_idx}")
            print(f"{'='*50}")
            
            # Create datasets
            train_dataset = BadNetDataset(df, train_participants, image_base_path, 
                                          label_type=label_type, transform=transform)
            val_dataset = BadNetDataset(df, val_participants, image_base_path, 
                                        label_type=label_type, transform=transform)
            test_dataset = BadNetDataset(df, test_participants, image_base_path, 
                                         label_type=label_type, transform=transform)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=4, pin_memory=True)
            
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            
            # Create model
            model = BadNetCNN(num_classes=num_classes, base_filters=base_filters, 
                             kernel_size=kernel_size, activation=activation)
            model = model.to(device)
            
            print("\nModel Architecture:")
            print(model)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                              factor=0.5, patience=10, verbose=True)
            
            # Train
            checkpoint_dir = f'./checkpoints/fold_{fold_idx}'
            history = train_fold(model, train_loader, val_loader, criterion, optimizer, 
                                scheduler, device, epochs, patience=20, checkpoint_dir=checkpoint_dir)
            
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
            
            # Evaluate on test set
            print(f"\nEvaluating Fold {fold_idx} on test set...")
            metrics = evaluate_model(model, test_loader, device, num_classes=num_classes)
            
            wandb.log({
                f'fold_{fold_idx}_test_accuracy': metrics['accuracy'],
                f'fold_{fold_idx}_test_precision': metrics['precision'],
                f'fold_{fold_idx}_test_recall': metrics['recall'],
                f'fold_{fold_idx}_test_f1': metrics['f1'],
                f'fold_{fold_idx}_test_kappa': metrics['kappa']
            })
            
            all_test_metrics.append(metrics)
            print(f"Fold {fold_idx} Test Metrics: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # Calculate average metrics across folds
        avg_accuracy = np.mean([m['accuracy'] for m in all_test_metrics])
        avg_precision = np.mean([m['precision'] for m in all_test_metrics])
        avg_recall = np.mean([m['recall'] for m in all_test_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_test_metrics])
        avg_kappa = np.mean([m['kappa'] for m in all_test_metrics])
        
        std_accuracy = np.std([m['accuracy'] for m in all_test_metrics])
        std_f1 = np.std([m['f1'] for m in all_test_metrics])
        
        print(f"\n{'='*50}")
        print("AVERAGE METRICS ACROSS ALL FOLDS")
        print(f"{'='*50}")
        print(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Cohen's Kappa: {avg_kappa:.4f}")
        
        # Log average metrics to W&B
        wandb.log({
            'avg_test_accuracy': avg_accuracy,
            'avg_test_precision': avg_precision,
            'avg_test_recall': avg_recall,
            'avg_test_f1': avg_f1,
            'avg_test_kappa': avg_kappa,
            'std_test_accuracy': std_accuracy,
            'std_test_f1': std_f1
        })
        
        wandb.run.summary.update({
            'avg_test_accuracy': avg_accuracy,
            'avg_test_precision': avg_precision,
            'avg_test_recall': avg_recall,
            'avg_test_f1': avg_f1,
            'avg_test_kappa': avg_kappa
        })
        
        wandb.finish()
    
    return main_func


def main():
    """Main function to run W&B sweep."""
    
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
            "csv_path": {"values": ['../../preprocessing/full_features/all_participants_0_3.csv']},
            "image_base_path": {"values": ['../../data/frames']},
            "exclude_participants": {"values": [[]]},  # Empty list, add participants to exclude
        },
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="BADNet_PyTorch")
    
    print(f"Sweep ID: {sweep_id}")
    print("Starting sweep agent...")
    
    # Run sweep
    wrapper_func = main_wrapper()
    wandb.agent(sweep_id, function=wrapper_func)


if __name__ == "__main__":
    main()
