import wandb
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from create_data_splits import create_data_splits
from get_metrics import get_test_metrics  # Assuming this function can be reused
import os
import math
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For synchronous CUDA errors
#Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # For CUDA device-side assertions

# Set seeds for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_tensor(tensor, name):
    if tensor.isnan().any() or tensor.isinf().any():
        print(f"Warning: {name} contains NaN or Inf values")
    print(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")


def prepare_targets(targets, label_type):
    """Process targets based on whether we're doing binary or multiclass classification."""
    if label_type == "binary_label":
        # For binary classification - ensure values are between 0 and 1
        if isinstance(targets[0], (list, np.ndarray)) and len(targets[0]) > 1:
            # If one-hot encoded, take first column as binary target
            targets = np.array([t[0] for t in targets])
        return torch.FloatTensor(targets).view(-1)
    else:
        # For multiclass classification - convert to class indices
        if isinstance(targets[0], (list, np.ndarray)) and len(targets[0]) > 1:
            # If one-hot encoded, convert to class indices
            targets = np.array([np.argmax(t) for t in targets])
        return torch.LongTensor(targets).view(-1)


class SequenceDataset(Dataset):
    def __init__(self, features, targets, label_type="binary_label"):
        self.features = torch.FloatTensor(features)
        self.targets = prepare_targets(targets, label_type)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MultiModalDataset(Dataset):
    def __init__(self, feature_sets, targets, label_type="binary_label"):
        self.feature_sets = [torch.FloatTensor(features) for features in feature_sets]
        self.targets = prepare_targets(targets, label_type)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return [features[idx] for features in self.feature_sets], self.targets[idx]

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, 
                                attn_mask=src_mask, 
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Handle both even and odd dimensions
        div_term_dim = d_model if d_model % 2 == 0 else d_model - 1
        div_term = torch.exp(torch.arange(0, div_term_dim, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin and cos to respective positions
        if d_model % 2 == 0:  # Even dimension
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:  # Odd dimension
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # One fewer cos term for odd dimensions
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Single Modality Transformer Model
class SingleModalityTransformer(nn.Module):
    def __init__(self, input_dims, seq_length, num_layers=2, nhead=8, dim_feedforward=2048, 
                 dropout=0.1, dense_units=128, activation='relu', num_classes=1):
        super(SingleModalityTransformer, self).__init__()
        
        # Explicitly calculate a multiple of nhead
        self.d_model = nhead * math.ceil(input_dims / nhead)
        
        # Verify divisibility
        assert self.d_model % nhead == 0, f"d_model ({self.d_model}) must be divisible by nhead ({nhead})"
        
        print(f"SingleModalityTransformer: input_dim={input_dims}, d_model={self.d_model}")
        
        # Input projection
        self.input_projection = nn.Linear(input_dims, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Create transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(self.d_model, dense_units)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling (take mean across sequence dimension)
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

# Early Fusion Transformer
class EarlyFusionTransformer(nn.Module):
    def __init__(self, input_dims, seq_length, num_layers=2, nhead=8, dim_feedforward=2048, 
                 dropout=0.1, dense_units=128, activation='relu',num_classes=1):
        super(EarlyFusionTransformer, self).__init__()
        
        # Since this is early fusion, we accept concatenated features directly
        self.model = SingleModalityTransformer(
            input_dims=input_dims,
            seq_length=seq_length,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dense_units=dense_units,
            activation=activation
        )
    
    def forward(self, x):
        return self.model(x)

# Intermediate Fusion Transformer
class IntermediateFusionTransformer(nn.Module):
    def __init__(self, input_dims, seq_length, num_layers=2, nhead=8, dim_feedforward=2048,
                 dropout=0.1, dense_units=128, activation='relu',num_classes=1):
        super(IntermediateFusionTransformer, self).__init__()
        
        # Create encoder for each modality
        self.modality_encoders = nn.ModuleList()
        self.projected_dims = []
        
        for idx, input_dim in enumerate(input_dims):
            # Explicitly calculate a dimension that's a multiple of nhead
            # Round up to nearest multiple of nhead
            d_model = nhead * math.ceil(input_dim / nhead)
            
            print(f"Modality {idx}: input_dim={input_dim}, d_model={d_model}")
            
            # Verify divisibility
            assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            
            self.projected_dims.append(d_model)
            
            # Create the encoder components
            projection = nn.Linear(input_dim, d_model)
            pos_encoder = PositionalEncoding(d_model)
            
            # Create transformer encoder with explicit dimension checks
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers-1)
            
            # Combine into a sequential module
            encoder = nn.Sequential(
                projection,
                pos_encoder,
                transformer
            )
            
            self.modality_encoders.append(encoder)
        
        # Calculate fusion dimension (sum of all projected dimensions)
        self.fusion_dim = sum(self.projected_dims)
        
        # Create a fusion dimension that's a multiple of nhead
        fusion_d_model = nhead * math.ceil(dense_units * 2 / nhead)
        
        # Verify divisibility
        assert fusion_d_model % nhead == 0, f"fusion_d_model ({fusion_d_model}) must be divisible by nhead ({nhead})"
        
        # Linear projection for fusion
        self.linear_projection = nn.Linear(self.fusion_dim, fusion_d_model)
        
        # Fusion transformer layer
        fusion_transformer_layer = nn.TransformerEncoderLayer(
            d_model=fusion_d_model,
            nhead=nhead,
            dim_feedforward=fusion_d_model*2,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_transformer_layer, num_layers=1)
        
        # Output layers
        self.fc1 = nn.Linear(fusion_d_model, dense_units)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_list):
        # Process each modality
        encoded_features = []
        
        for i, encoder in enumerate(self.modality_encoders):
            features = encoder(x_list[i])
            encoded_features.append(features)
        
        # Concatenate along feature dimension
        x = torch.cat(encoded_features, dim=2)
        
        # Project to common dimension
        x = self.linear_projection(x)
        
        # Apply fusion transformer
        x = self.fusion_encoder(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

# Late Fusion Transformer
class LateFusionTransformer(nn.Module):
    def __init__(self, input_dims, seq_length, num_layers=2, nhead=8, dim_feedforward=2048,
                 dropout=0.1, dense_units=128, activation='relu',num_classes=1):
        super(LateFusionTransformer, self).__init__()
        
        # Create separate transformers for each modality
        self.modality_transformers = nn.ModuleList()
        
        for idx, input_dim in enumerate(input_dims):
            print(f"Creating transformer for modality {idx} with input_dim={input_dim}")
            transformer = SingleModalityTransformer(
                input_dims=input_dim,
                seq_length=seq_length,
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                dense_units=dense_units,
                activation=activation
            )
            self.modality_transformers.append(transformer)
        
        # Fusion dimension based on dense_units from each modality
        self.fusion_dim = dense_units * len(input_dims)
        
        # Fusion layers
        self.fusion_fc = nn.Linear(self.fusion_dim, dense_units)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_list):
        # Process each modality but extract features before the final sigmoid
        modality_features = []
        
        for i, transformer in enumerate(self.modality_transformers):
            # Extract the model from the transformer
            model = transformer
            
            # Project input to appropriate dimension
            x = model.input_projection(x_list[i])
            
            # Apply positional encoding
            x = model.pos_encoder(x)
            
            # Pass through transformer encoder
            x = model.transformer_encoder(x)
            
            # Global pooling
            x = torch.mean(x, dim=1)
            
            # Get features from first dense layer
            x = model.fc1(x)
            x = model.activation(x)
            
            modality_features.append(x)
        
        # Concatenate features from all modalities
        x = torch.cat(modality_features, dim=1)
        
        # Apply fusion layers
        x = self.fusion_fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_fc(x)
        x = self.sigmoid(x)
        
        return x

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=3, label_type="binary_label"):
    # Try using CPU if CUDA issues persist
    

    if label_type == "binary_label":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    try:
        model.to(device)
    except RuntimeError:
        print("CUDA error encountered. Falling back to CPU...")
        device = torch.device("cpu")
        model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                if isinstance(data, list):
                    # For multi-modal data
                    data = [d.to(device) for d in data]
                else:
                    data = data.to(device)
                
                # Ensure target is proper shape and type
                target = target.to(device).float()
                if target.dim() == 0:
                    target = target.unsqueeze(0)  # Handle scalar case
                
                optimizer.zero_grad()
                output = model(data)
                
                # Process output based on task
                if label_type == "binary_label":
                    # Binary classification with BCE loss
                    if output.dim() > 1 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    
                    # Safety check for binary targets
                    if torch.any(target < 0) or torch.any(target > 1):
                        print(f"Warning: Binary targets outside [0,1] range, clamping...")
                        target = torch.clamp(target, 0, 1)
                else:
                    # No special handling needed for multiclass with CrossEntropyLoss
                    pass
                    
                # Safety check for NaN values
                if torch.isnan(output).any():
                    print(f"Warning: NaN in output at batch {batch_idx}")
                    continue
                
                # Ensure dimensions match
                if output.shape != target.shape:
                    #print(f"Shape mismatch: output {output.shape}, target {target.shape}")
                    if output.numel() == target.numel():
                        output = output.view(target.shape)
                
                loss = criterion(output, target)
                
                # Check for NaN in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: Loss is NaN/Inf at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = (output > 0.5).float()
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
                
            except RuntimeError as e:
                #print(f"Error in batch {batch_idx}: {e}")
                #if "CUDA" in str(e):
                #    print("CUDA error, trying to recover...")
                #    torch.cuda.empty_cache()
                #    continue
                #else:
                #    raise e
                raise e
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    if isinstance(data, list):
                        # For multi-modal data
                        data = [d.to(device) for d in data]
                    else:
                        data = data.to(device)
                    
                    # Ensure target is proper shape and type
                    target = target.to(device).float()
                    if target.dim() == 0:
                        target = target.unsqueeze(0)  # Handle scalar case
                    
                    output = model(data)
                    
                    # Process output based on task
                    if label_type == "binary_label":
                        # Binary classification with BCE loss
                        if output.dim() > 1 and output.size(-1) == 1:
                            output = output.squeeze(-1)
                        
                        # Safety check for binary targets
                        if torch.any(target < 0) or torch.any(target > 1):
                            print(f"Warning: Binary targets outside [0,1] range, clamping...")
                            target = torch.clamp(target, 0, 1)
                    else:
                        # No special handling needed for multiclass with CrossEntropyLoss
                        pass
                    
                    # Ensure dimensions match
                    if output.shape != target.shape:
                        if output.numel() == target.numel():
                            output = output.view(target.shape)
                    
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = (output > 0.5).float()
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
                    
                except RuntimeError as e:
                    print(f"Error in validation: {e}")
                    if "CUDA" in str(e):
                        print("CUDA error, trying to recover...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Calculate metrics, avoiding division by zero
        if train_total > 0:
            train_accuracy = train_correct / train_total
        else:
            train_accuracy = 0.0
            
        if val_total > 0:
            val_accuracy = val_correct / val_total
        else:
            val_accuracy = 0.0
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss / max(1, len(train_loader)),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss / max(1, len(val_loader)),
            'val_accuracy': val_accuracy
        }
        
        wandb.log(metrics)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_accuracy']:.4f}, "
              f"Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_accuracy']:.4f}")
    
    return model

# Test function
def test_model(model, test_loader, device, label_type="binary_label"):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, list):
                # For multi-modal data
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            
            target = target.to(device).float() if label_type == "binary_label" else target.to(device).long()
            
            output = model(data)
            
            if label_type == "binary_label":
                # Binary classification with BCE loss
                if output.dim() > 1 and output.size(-1) == 1:
                    output = output.squeeze(-1)
                
                # Safety check for binary targets
                if torch.any(target < 0) or torch.any(target > 1):
                    print(f"Warning: Binary targets outside [0,1] range, clamping...")
                    target = torch.clamp(target, 0, 1)
                
                pred = (output > 0.5).float()
            else:
                # Multiclass classification with CrossEntropyLoss
                pred = torch.argmax(output, dim=-1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets)


def create_model_for_task(model_class, label_type, *args, **kwargs):
    """Create and modify a model based on the classification task."""
    model = model_class(*args, **kwargs)
    
    # For multiclass, replace the final sigmoid with a softmax layer
    if label_type != "binary_label":
        # Get number of classes (assuming it's passed in kwargs)
        num_classes = kwargs.get('num_classes', 2)  # Default to 2 if not specified
        
        # Replace the final layers
        if hasattr(model, 'fc2'):
            model.fc2 = nn.Linear(model.fc2.in_features, num_classes)
        
        # Remove sigmoid activation for multiclass (we'll use softmax with CrossEntropyLoss)
        if hasattr(model, 'sigmoid'):
            model.sigmoid = nn.Identity()
    
    return model

# Early Fusion Training Function
def train_early_fusion(df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    num_layers = config.num_layers
    nhead = config.nhead
    batch_size = config.batch_size
    epochs = config.epochs
    activation = config.activation_function
    dropout = config.dropout_rate
    optimizer_name = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
    sequence_length = config.sequence_length

    
    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }
    
    for fold in range(5):
        print(f"Fold {fold}")
        
        splits = create_data_splits(
            df, config.label,
            fold_no=fold,
            num_folds=5,
            seed_value=42,
            sequence_length=sequence_length
        )
        
        if splits is None:
            continue
            
        X_train, X_val, X_test, y_train, y_val, y_test, \
        X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, \
        X_test_sequences, y_test_sequences, sequence_length = splits
        
        input_dim = X_train_sequences.shape[2]
        
        # Create datasets and dataloaders
        train_dataset = SequenceDataset(X_train_sequences, y_train_sequences, label_type=config.label)
        val_dataset = SequenceDataset(X_val_sequences, y_val_sequences, label_type=config.label)
        test_dataset = SequenceDataset(X_test_sequences, y_test_sequences, label_type=config.label)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Get number of classes for multiclass tasks
        if config.label != "binary_label":
            # Calculate number of classes from the data
            num_classes = len(np.unique(y_train_sequences))
            print(f"Multiclass classification with {num_classes} classes")
        else:
            num_classes = 1

        # Create model with appropriate task configuration
        model = create_model_for_task(
            EarlyFusionTransformer,  # or whichever model class you're using
            config.label,
            input_dims=input_dim,
            seq_length=sequence_length,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dense_units * 4,
            dropout=dropout,
            dense_units=dense_units,
            activation=activation,
            num_classes=num_classes
        )

        
                
        # Setup optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function
        if config.label != "binary_label":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        
        # Train model
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            label_type=config.label
        )

        
        # Test model
        y_pred, y_true = test_model(model, test_loader, device, label_type=config.label)
        
        # Get metrics
        test_metrics = get_test_metrics(y_pred, y_true, tolerance=1)
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])
        
        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
    # Calculate average metrics
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)

# Intermediate Fusion Training Function
def train_intermediate_fusion(modality_dfs, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    num_layers = config.num_layers
    nhead = config.nhead
    batch_size = config.batch_size
    epochs = config.epochs
    activation = config.activation_function
    dropout = config.dropout_rate
    optimizer_name = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
    sequence_length = config.sequence_length
    
    modality_keys = list(modality_dfs.keys())
    
    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }
    
    for fold in range(5):
        print(f"Fold {fold}")
        
        splits = {}
        for modality_key in modality_keys:
            df = modality_dfs[modality_key]
            splits[modality_key] = create_data_splits(
                df, config.label,
                fold_no=fold,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length
            )
            
            if splits[modality_key] is None:
                print(f"Skipping fold {fold} due to empty split for modality {modality_key}")
                continue
        
        # Get training sequences for each modality
        train_sequences = [splits[m][6] for m in modality_keys]
        val_sequences = [splits[m][8] for m in modality_keys]
        test_sequences = [splits[m][10] for m in modality_keys]
        
        # Get targets (same for all modalities)
        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7]
        y_val_sequences = splits[first_modality][9]
        y_test_sequences = splits[first_modality][11]
        
        # Create datasets and dataloaders
        train_dataset = MultiModalDataset(train_sequences, y_train_sequences)
        val_dataset = MultiModalDataset(val_sequences, y_val_sequences)
        test_dataset = MultiModalDataset(test_sequences, y_test_sequences)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Get input dimensions for each modality
        input_dims = [seq.shape[2] for seq in train_sequences]
        
        if config.label != "binary_label":
            # Calculate number of classes from the data
            num_classes = len(np.unique(y_train_sequences))
            print(f"Multiclass classification with {num_classes} classes")
        else:
            num_classes = 1

        # Create model with appropriate task configuration
        model = create_model_for_task(
            IntermediateFusionTransformer,  # or whichever model class you're using
            config.label,
            input_dims=input_dims,
            seq_length=sequence_length,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dense_units * 4,
            dropout=dropout,
            dense_units=dense_units,
            activation=activation,
            num_classes=num_classes
        )

        
       
        
        # Setup optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function
        if config.label != "binary_label":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        
        # Train model
        # Train with appropriate loss function
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            label_type=config.label,
            criterion=criterion
        )
        # Test model
        y_pred, y_true = test_model(model, test_loader, device, label_type=config.label)
        
        # Get metrics
        test_metrics = get_test_metrics(y_pred, y_true, tolerance=1)
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])
        
        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
    # Calculate average metrics
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)

# Late Fusion Training Function
def train_late_fusion(modality_dfs, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    num_layers = config.num_layers
    nhead = config.nhead
    batch_size = config.batch_size
    epochs = config.epochs
    activation = config.activation_function
    dropout = config.dropout_rate
    optimizer_name = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
    sequence_length = config.sequence_length
    
    modality_keys = list(modality_dfs.keys())
    
    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }
    
    for fold in range(5):
        print(f"Fold {fold}")
        
        splits = {}
        valid_modalities = []
        
        # First, collect splits for each modality and track which ones are valid
        for modality_key in modality_keys:
            df = modality_dfs[modality_key]
            result = create_data_splits(
                df, config.label,
                fold_no=fold,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length
            )
            
            if result is not None:
                splits[modality_key] = result
                valid_modalities.append(modality_key)
            else:
                print(f"Warning: Skipping modality {modality_key} for fold {fold} due to None splits")
        
        # Skip this fold if no valid modalities
        if not valid_modalities:
            print(f"Skipping fold {fold} completely - no valid modalities")
            continue
            
        # Continue with only valid modalities
        #try:
        # Get training sequences for each valid modality
        train_sequences = [splits[m][6] for m in valid_modalities]
        val_sequences = [splits[m][8] for m in valid_modalities]
        test_sequences = [splits[m][10] for m in valid_modalities]
        
        # Get targets (same for all modalities) - use first valid modality
        first_modality = valid_modalities[0]
        y_train_sequences = splits[first_modality][7]
        y_val_sequences = splits[first_modality][9]
        y_test_sequences = splits[first_modality][11]
        
        # Create datasets and dataloaders
        train_dataset = MultiModalDataset(train_sequences, y_train_sequences)
        val_dataset = MultiModalDataset(val_sequences, y_val_sequences)
        test_dataset = MultiModalDataset(test_sequences, y_test_sequences)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Get input dimensions for each modality
        input_dims = [seq.shape[2] for seq in train_sequences]
        print(y_train_sequences.shape)
        print('above, shape')

        if config.label != "binary_label":
            # Calculate number of classes from the data
            num_classes = len(np.unique(y_train_sequences))
            print(f"Multiclass classification with {num_classes} classes")
        else:
            num_classes = 1
        
        # Create model with appropriate task configuration
        model = create_model_for_task(
            LateFusionTransformer,  # or whichever model class you're using
            config.label,
            input_dims=input_dims,
            seq_length=sequence_length,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dense_units * 4,
            dropout=dropout,
            dense_units=dense_units,
            activation=activation,
        )

        
       
        
        # Setup optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function
        if config.label != "binary_label":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        
        # Train model
        # Train with appropriate loss function
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            label_type=config.label,
            criterion=criterion
        )
        
        # Test model
        y_pred, y_true = test_model(model, test_loader, device, label_type=config.label)
        
        # Get metrics
        test_metrics = get_test_metrics(y_pred, y_true, tolerance=1)
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])
        
        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
    # Calculate average metrics
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)

# Main training function
def train():
    wandb.init()
    config = wandb.config
    print(config)
    
    set_seed(42)
    
    feature_set = config.feature_set
    modality = config.modality
    data = config.data
    fusion_type = config.fusion_type
    
    # Data loading (same as original script)
    df = pd.read_csv("../../data/raw/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../data/feature_sets/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../data/feature_sets/all_participants_rf_0_3_40.csv")
    df_text = pd.read_csv("../../preprocessing/text_embeddings.csv")
    df_text_pca = pd.read_csv("../../data/embeddings/text_embeddings_pca.csv")
    df_text_clip = pd.read_csv("../../preprocessing/clip_text_embeddings.csv")
    df_text_clip_pca = pd.read_csv("../../data/embeddings/clip_text_embeddings_pca.csv")
    
    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)  # action units, gaze
    df_audio_index = df.iloc[:, 63:88]
    df_text_index = df_text.iloc[:, 2:]
    df_text_pca_index = df_text_pca.iloc[:, 2:]
    df_text_clip_index = df_text_clip.iloc[:, 2:]
    df_text_clip_pca_index = df_text_clip_pca.iloc[:, 2:]


    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]
    
    df_facial_index_rf = df_rf.iloc[:, 38:]
    df_pose_index_rf = df_rf.iloc[:, 4:28]
    df_audio_index_rf = df_rf.iloc[:, 28:38]
    
    print(f"--------\nfeature set: {feature_set}\nmodality: {modality}\ndata: {data}\nfusion: {fusion_type}\n--------")
    
    # Data preparation functions (same as original)
    def create_normalized_df(df):
        if df.empty:
            raise ValueError("create_normalized_df: Input DataFrame is empty.")
        participant_frames_labels = df.iloc[:, :4]
        
        features = df.columns[4:]
        norm_df = df.copy()
        
        scaler = StandardScaler()
        norm_df[features] = scaler.fit_transform(norm_df[features])
        
        norm_df = pd.concat([participant_frames_labels, norm_df[features]], axis=1)
        
        return norm_df
    
    def create_pca_df(df):
        participant_frames_labels = df.iloc[:, :4]
        
        x = df.iloc[:, 4:]
        x = StandardScaler().fit_transform(x.values)
        
        pca = PCA(n_components=0.90)
        principal_components = pca.fit_transform(x)
        print(principal_components.shape)
        
        principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
        principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)
        
        return principal_df
    
    # Define modalities dictionary (same as original)
    modalities = {
        "pose_full": df_pose_index,
        "pose_rf": df_pose_index_rf,
        
        "facial_full": df_facial_index,
        "facial_stats": df_facial_index_stats,
        "facial_rf": df_facial_index_rf,
        
        "audio_full": df_audio_index,
        "audio_stats": df_audio_index_stats,
        "audio_rf": df_audio_index_rf,
        
        "text": df_text_index,
        "textpca": df_text_pca_index,
        "textclip": df_text_clip_index,
        "textclippca": df_text_clip_pca_index


    }
    
    # Select modalities based on configuration (same as original)
    modality_components = modality.split('_')
    selected_modalities = {}
    
    if "pose" in modality_components:
        if feature_set == "full":
            selected_modalities["pose_full"] = modalities["pose_full"]
        elif feature_set == "rf":
            selected_modalities["pose_rf"] = modalities["pose_rf"]
    
    if "facial" in modality_components:
        if feature_set == "full":
            selected_modalities["facial_full"] = modalities["facial_full"]
        elif feature_set == "stats":
            selected_modalities["facial_stats"] = modalities["facial_stats"]
        elif feature_set == "rf":
            selected_modalities["facial_rf"] = modalities["facial_rf"]
    
    if "audio" in modality_components:
        if feature_set == "full":
            selected_modalities["audio_full"] = modalities["audio_full"]
        elif feature_set == "stats":
            selected_modalities["audio_stats"] = modalities["audio_stats"]
        elif feature_set == "rf":
            selected_modalities["audio_rf"] = modalities["audio_rf"]
    
    if "text" in modality_components:
        if "clip" in modality_components:
            if "pca" in modality_components:
                selected_modalities["textclippca"] = modalities["textclippca"]
            else:
                selected_modalities["textclip"] = modalities["textclip"]
        else:
            if "pca" in modality_components:
                selected_modalities["textpca"] = modalities["textpca"]
            else:
                selected_modalities["text"] = modalities["text"]
    
    # Train based on fusion type
    if fusion_type == "early":
        df = info
        print(df)
        print(df.shape)
        print(selected_modalities)
        print('adding modalities')
        for m in selected_modalities.values():
            print('shapes before adding')
            print(df.shape)
            print(m.shape)
            df = pd.concat([df, m], axis=1)
            print('new_df_size')
            print(df)
        
        if data == "norm":
            df = create_normalized_df(df)
        elif data == "pca":
            df = create_pca_df(create_normalized_df(df))
        
        print(df)
        print(df.shape)
        
        train_early_fusion(df, config)
    
    elif fusion_type == "intermediate" or fusion_type == "late":
        dfs = {}
        
        if data == "norm":
            for modality_name, m in selected_modalities.items():
                df_temp = pd.concat([info.copy(), m], axis=1)
                dfs[modality_name] = create_normalized_df(df_temp)
        elif data == "pca":
            for modality_name, m in selected_modalities.items():
                if modality_name == "textpca":
                    dfs[modality_name] = df_text_pca
                else:
                    df_temp = pd.concat([info.copy(), m], axis=1)
                    dfs[modality_name] = create_pca_df(create_normalized_df(df_temp))
        elif data == "reg":
            for modality_name, m in selected_modalities.items():
                df_temp = pd.concat([info.copy(), m], axis=1)
                dfs[modality_name] = df_temp
        
        print(dfs)
        
        if fusion_type == "intermediate":
            train_intermediate_fusion(dfs, config)
        elif fusion_type == "late":
            train_late_fusion(dfs, config)

        #end sweep

        wandb.finish()
        #clear cache and others for saving memory

        torch.cuda.empty_cache()
        


# Main function - setup wandb sweep
def main():
    feature_set = random.choice(["full", "stats", "rf"])
    #TEMPPPPPPPPPPPPPPPPP
    feature_set = "full"
    
    if feature_set == "full":
        modality = random.choice(['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio', 
        'text','textpca', 'textclip','textclippca','pose_facial_audio_text', 'pose_facial_audio_textpca', 'pose_facial_text', 'pose_facial_textpca', 'pose_audio_text', 'pose_audio_textpca',
        'facial_audio_text', 'facial_audio_textpca', 'pose_text', 'pose_textpca', 'facial_text', 'facial_textpca', 'audio_text', 'audio_textpca',
        'pose_facial_textclip', 'pose_facial_textclippca', 'pose_audio_textclip', 'pose_audio_textclippca',
        'facial_audio_textclip', 'facial_audio_textclippca', 'pose_textclip', 'pose_textclippca', 'facial_textclip', 'facial_textclippca',
         'audio_textclip', 'audio_textclippca','pose_facial_audio_textclip', 'pose_facial_audio_textclippca'])
    elif feature_set == "stats":
        modality = random.choice(['facial_audio', 'facial', 'audio'])
    elif feature_set == "rf":
        modality = random.choice(['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio'])

    
    sweep_config = {
        'method': 'random',
        'name': 'transformer_all',
        'parameters': {
            'feature_set': {'values': [feature_set]},
            'modality': {'values': [modality]},
            
            'data': {'values': ["reg", "norm", "pca"]},
            'fusion_type': {'values': ['early', 'intermediate', 'late']},
            
            # Transformer-specific parameters
            'num_layers': {'values': [1, 2, 4, 6]},
            'nhead': {'values': [4, 8]},
            'dropout_rate': {'values': [0.1, 0.3, 0.5]},
            'dense_units': {'values': [64, 128, 256]},
            'activation_function': {'values': ['relu', 'tanh']},
            'optimizer': {'values': ['adam', 'sgd', 'rmsprop']},
            'learning_rate': {'values': [0.0001, 0.001, 0.01]},
            'batch_size': {'values': [32, 64,128]},
            'epochs': {'value': 100},
            
            'sequence_length': {'values': [5,15,30,80]},
            'label': {'values': ['binary_label', 'multiclass_label']}, #'binary_label', 'multiclass_label'
        }
    }
    
    print(sweep_config)
    
    
    def train_wrapper():
        train()
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="transformer_all")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()