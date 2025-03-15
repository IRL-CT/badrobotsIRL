import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import wandb
import argparse

import sys
import os
import wget


# Set memory management environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Clone AST repository if not exists
if not os.path.exists('ast'):
    os.system('git clone https://github.com/YuanGongND/ast')
sys.path.append('./ast')


# Import AST model
from src.models import ASTModel


#from src.models import ASTModel
from ast_feats import extract_session_features
from get_metrics import get_metrics
from create_data_splits import create_data_splits_folds

# Define sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': -9.21,  # ln(0.0001)
            'max': -4.61   # ln(0.01)
        },
        'weight_decay': {
            'distribution': 'log_uniform',
            'min': -9.21,  # ln(0.0001)
            'max': -2.30   # ln(0.1)
        },
        'batch_size': {
            'values': [32]
        },
        'gradient_accumulation_steps': {
            'values': [4, 8]  # Add gradient accumulation
        },
        'num_epochs': {
            'value': 100
        }
    }
}

class ASTDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


        
class ASTFineTuner:
    def __init__(self, num_classes: int, config: dict, device: str = 'cuda'):
        self.device = device
        self.config = config
        self.model = self._load_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        # Only get parameters that require gradients
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


    def _load_model(self, num_classes: int) -> nn.Module:
        model = ASTModel(
            label_dim=num_classes,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False
        )

        audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
        if not os.path.exists('./pretrained_ast/audio_mdl.pth'):
            os.makedirs('./pretrained_ast', exist_ok=True)
            wget.download(audioset_mdl_url, out='./pretrained_ast/audio_mdl.pth')

        checkpoint_path = './pretrained_ast/audio_mdl.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'module.v.head.weight' in checkpoint:
            del checkpoint['module.v.head.weight']
            del checkpoint['module.v.head.bias']
        
        model.load_state_dict(checkpoint, strict=False)

        # Print parameter status before and after freezing
        print("\nParameter gradients status:")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
        
        # Freeze base model parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze the head
        for param in model.v.head.parameters():
            param.requires_grad = True

        
            
        # Make sure the head is properly initialized
        model.v.head.weight.requires_grad = True
        model.v.head.bias.requires_grad = True
        model.v.head_dist.weight.requires_grad = True
        model.v.head_dist.bias.requires_grad = True
        model.mlp_head[0].weight.requires_grad = True
        model.mlp_head[1].weight.requires_grad = True
        model.mlp_head[0].bias.requires_grad = True
        model.mlp_head[1].bias.requires_grad = True

        # Print parameter status before and after freezing
        print("\nParameter gradients status:")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
        

        # Print final status of head parameters
        print("\nHead parameters status:")
        for name, param in model.v.head.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
            print(f"Parameter shape: {param.shape}")

        return model.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[ float, Dict]:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Clear GPU cache at the start of each epoch
        torch.cuda.empty_cache()
        
        pbar = tqdm(train_loader, desc='Training')
        self.optimizer.zero_grad()  # Move outside the loop
        
        for i, (features, labels) in enumerate(pbar):
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Scale loss by gradient accumulation steps
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Only step optimizer after accumulating gradients
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            
            # Move predictions to CPU to save GPU memory
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            pbar.set_postfix({'loss': total_loss/len(train_loader)})
        
        # Final optimizer step for remaining gradients
        if len(train_loader) % self.config.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        # Calculate metrics
        metrics = get_metrics(np.array(all_labels), np.array(all_preds)) ##########BE CAREFUL WITH ypred and ytrue, check the order in script
        #print count of each class
        print('Train metrics:', metrics)
        all_labels_count = np.unique(all_labels, return_counts=True)
        print('Train labels count:', all_labels_count)
        all_preds_count = np.unique(all_preds, return_counts=True)
        print('Train preds count:', all_preds_count)

        return total_loss / len(train_loader), metrics


    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = get_metrics(all_labels, all_preds)
        return total_loss / len(val_loader), metrics




def process_fold_data(audio_files: List[str], label_df: pd.DataFrame, audio_dir: Path):
    """Process audio files for a specific fold"""
    fold_features = []
    fold_labels = []
    
    
    for audio_file in audio_files:
        torch.cuda.empty_cache()
        audio_path = audio_dir / f"{audio_file}.wav"
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue
            
        # Get labels for this participant
        participant_df = label_df[label_df['participant'] == audio_file]
        if participant_df.empty:
            print(f"Warning: No labels found for participant {audio_file}")
            continue
            
        features = extract_session_features(str(audio_path), participant_df)
        #print(features.shape)
        if features is None:
            print(f"Warning: No features found for participant {audio_file}")
        labels = participant_df['binary_label'].values
        
        fold_features.append(features)
        fold_labels.append(labels)
    #stack lists into tensors
    #fold_labels = [torch.tensor(l) for l in fold_labels]
    #collapse labels
    fold_labels = [item for sublist in fold_labels for item in sublist]
    #do the same for features
    fold_features = [item for sublist in fold_features for item in sublist]
    fold_features = torch.stack(fold_features)
    print(fold_features.shape)


    return fold_features, fold_labels

def train():
    # Initialize wandb
    wandb.init()
    config = wandb.config
    
    # Set seeds for reproducibility
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Default arguments
    audio_dir = Path('nodbot_wav_files_silenced')
    label_path = 'all_participants_0_3.csv'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, default=str(audio_dir))
    parser.add_argument('--label_path', type=str, default=label_path)
    #parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load all labels
    label_df = pd.read_csv(args.label_path)
    audio_dir = Path(args.audio_dir)
    #label_df = args.label_path
    #output_dir = Path(args.output_dir)
    #output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics storage for final epoch
    final_metrics = {
        'train': [None] * 5,
        'val': [None] * 5,
        'test': [None] * 5
    }
    
    # Perform 5-fold cross-validation
    for fold in range(5):
        print(f"\nProcessing fold {fold + 1}/5")
        
        # Get train/val/test splits for this fold
        train_sessions, val_sessions, test_sessions = create_data_splits_folds(
            label_df, num_folds=5, fold_no=fold, 
            seed_value=seed_value
        )
        
        # Process data for this fold
        train_features, train_labels = process_fold_data(train_sessions, label_df, audio_dir)
        val_features, val_labels = process_fold_data(val_sessions, label_df, audio_dir)
        test_features, test_labels = process_fold_data(test_sessions, label_df, audio_dir)
        
        # Create datasets and dataloaders
        train_dataset = ASTDataset(train_features, train_labels)
        val_dataset = ASTDataset(val_features, val_labels)
        test_dataset = ASTDataset(test_features, test_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,  # Add pin_memory
            num_workers=1     # Add num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size
        )
        
        # Initialize model for this fold
        fine_tuner = ASTFineTuner(
            num_classes=len(set(label_df['binary_label'])),
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Training loop for this fold
        for epoch in range(config.num_epochs):
            print(f'Fold {fold + 1}, Epoch {epoch + 1}/{config.num_epochs}')
            
            train_loss, train_metrics = fine_tuner.train_epoch(train_loader)
            val_loss, val_metrics = fine_tuner.validate(val_loader)
            
            # Log fold-specific metrics
            metrics_dict = {
                f'{fold}_train_loss': train_loss,
                f'{fold}_val_loss': val_loss,
                **{f'{fold}_train_{k}': v for k, v in train_metrics.items()},
                **{f'{fold}_val_{k}': v for k, v in val_metrics.items()}
            }
            
            wandb.log(metrics_dict)
            
            # Store final epoch metrics
            if epoch == config.num_epochs - 1:
                final_metrics['train'][fold] = train_metrics
                final_metrics['val'][fold] = val_metrics
        
        # Evaluate on test set after training is complete
        test_loss, test_metrics = fine_tuner.validate(test_loader)
        final_metrics['test'][fold] = test_metrics
        
        # Log test metrics for this fold
        test_metrics_dict = {
            f'{fold}_test_loss': test_loss,
            **{f'{fold}_test_{k}': v for k, v in test_metrics.items()}
        }
        wandb.log(test_metrics_dict)
    
    # Calculate and log average metrics for final epoch
    avg_metrics = {}
    for split in ['train', 'val', 'test']:
        for metric in final_metrics[split][0].keys():
            avg_value = np.mean([
                final_metrics[split][f][metric] 
                for f in range(5)
            ])
            avg_metrics[f'avg_{split}_{metric}'] = avg_value
    
    # Log average metrics
    wandb.log(avg_metrics)
            
            # Save checkpoint
            #torch.save({
            #    'fold': fold,
            #    'epoch': epoch,
            #    'model_state_dict': fine_tuner.model.state_dict(),
            #    'optimizer_state_dict': fine_tuner.optimizer.state_dict(),
            #    'train_metrics': train_metrics,
            #    'val_metrics': val_metrics,
            #    'config': config,
            #}, output_dir / f'checkpoint_fold_{fold}_epoch_{epoch + 1}.pt')

if __name__ == "__main__":
    # Initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="ast-finetuning")
    
    # Start the sweep
    wandb.agent(sweep_id, train)