#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BADNet Robot Behavior Classification - PyTorch Implementation

Core components: models, datasets, and training utilities.
"""

import os
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models


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
    """Dataset for loading robot behavior images with frame-to-label mapping."""
    
    def __init__(self, df, participants, image_base_path, label_type='binary', transform=None, cache_images=True):
        self.df = df[df['participant'].isin(participants)].reset_index(drop=True)
        self.image_base_path = image_base_path
        self.label_type = label_type
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            participant = row['participant']
            frame = int(row['frame'])
            
            if label_type == 'binary':
                label = int(row['binary_label'])
            else:
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


class BadNetDatasetWithAugmentation(Dataset):
    """Dataset that includes both original and augmented samples."""
    
    def __init__(self, df, participants, image_base_path, label_type='binary', num_augmentations=2):
        self.df = df[df['participant'].isin(participants)].reset_index(drop=True)
        self.image_base_path = image_base_path
        self.label_type = label_type
        self.num_augmentations = num_augmentations
        
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        
        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            participant = row['participant']
            frame = int(row['frame'])
            
            if label_type == 'binary':
                label = int(row['binary_label'])
            else:
                label = int(row['multiclass_label'])
            
            image_path = os.path.join(image_base_path, participant, f"{frame}.jpg")
            if os.path.exists(image_path):
                self.samples.append((image_path, label, False))
                for _ in range(num_augmentations):
                    self.samples.append((image_path, label, True))
        
        print(f"Dataset: {len(self.df)} original → {len(self.samples)} total samples "
              f"({num_augmentations} augmentations per image)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label, is_augmented = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if is_augmented:
            image = self.augmentation_transform(image)
        
        image = self.base_transform(image)
        return image, label


class BadNetCNN(nn.Module):
    """Original CNN model for robot behavior classification."""
    
    def __init__(self, num_classes=2, base_filters=16, kernel_size=4, activation='relu'):
        super(BadNetCNN, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        self.conv1 = nn.Conv2d(3, base_filters, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.dropout1 = nn.Dropout(0.15)
        
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.dropout3 = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm2d(base_filters*4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout4 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(base_filters*4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
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
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.batchnorm(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout4(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x


class BadNetPretrained(nn.Module):
    """Pretrained ResNet model for transfer learning."""
    
    def __init__(self, num_classes=2, backbone='resnet18', freeze_backbone=True, dropout=0.5):
        super(BadNetPretrained, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last few layers for fine-tuning
            if backbone.startswith('resnet'):
                for param in list(self.backbone.parameters())[-20:]:
                    param.requires_grad = True
            elif backbone.startswith('efficientnet'):
                for param in list(self.backbone.parameters())[-30:]:
                    param.requires_grad = True
        
        # Replace final classifier
        if backbone.startswith('resnet'):
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(256, num_classes)
            )
        elif backbone.startswith('efficientnet'):
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


class BadNetSimple(nn.Module):
    """Simpler/shallower CNN model."""
    
    def __init__(self, num_classes=2, base_filters=32, dropout=0.25):
        super(BadNetSimple, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, base_filters, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*2*16, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(model_type='original', num_classes=2, **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        model_type: 'original', 'simple', 'pretrained_resnet18', 'pretrained_resnet34', 
                    'pretrained_resnet50', 'pretrained_efficientnet_b0'
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Model instance
    """
    if model_type == 'original':
        return BadNetCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'simple':
        return BadNetSimple(num_classes=num_classes, **kwargs)
    elif model_type.startswith('pretrained_'):
        backbone = model_type.replace('pretrained_', '')
        return BadNetPretrained(num_classes=num_classes, backbone=backbone, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_interparticipant_folds(df, num_folds=5, exclude_participants=None, seed=42):
    """Create inter-participant folds for cross-validation."""
    np.random.seed(seed)
    
    all_participants = df['participant'].unique().tolist()
    
    if exclude_participants:
        all_participants = [p for p in all_participants if p not in exclude_participants]
        print(f"Excluded {len(exclude_participants)} participants. Remaining: {len(all_participants)}")
    
    np.random.shuffle(all_participants)
    
    fold_size = len(all_participants) // num_folds
    participant_groups = []
    
    for i in range(num_folds):
        if i == num_folds - 1:
            participant_groups.append(all_participants[i * fold_size:])
        else:
            participant_groups.append(all_participants[i * fold_size:(i + 1) * fold_size])
    
    folds = []
    for fold_idx in range(num_folds):
        test_participants = participant_groups[fold_idx]
        
        train_val_participants = []
        for i in range(num_folds):
            if i != fold_idx:
                train_val_participants.extend(participant_groups[i])
        
        np.random.shuffle(train_val_participants)
        
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


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, epochs, patience=20, checkpoint_dir='./checkpoints'):
    """Train model for one fold with early stopping."""
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(checkpoint_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history