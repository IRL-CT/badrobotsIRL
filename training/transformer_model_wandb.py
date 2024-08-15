import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb

from create_data_splits import create_data_splits
from get_metrics import get_metrics

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, num_classes, dropout, activation, loss_type):
        super(TransformerModel, self).__init__()
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout, 
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

        if loss_type == "binary_crossentropy":
            self.output_activation = nn.Sigmoid()
        elif loss_type == "categorical_crossentropy":
            self.output_activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.output_activation(x)
        return x

def train(df):
    
    wandb.init()
    config = wandb.config
    print(config)

    num_heads = config.num_heads
    hidden_dim = config.hidden_dim
    num_layers = config.num_layers
    batch_size = config.batch_size
    activation = config.activation_function
    dropout = config.dropout_rate
    optimizer_type = config.optimizer
    learning_rate = config.learning_rate
    loss_type = config.loss
    sequence_length = config.sequence_length
    output_dim = 1
    
    splits = create_data_splits(
        df,
        fold_no=0,
        num_folds=5,
        seed_value=42,
        sequence_length=sequence_length
    )
    
    if splits is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

    print("X_train_sequences shape:", X_train_sequences.shape)
    print("X_val_sequences shape:", X_val_sequences.shape)
    print("X_test_sequences shape:", X_test_sequences.shape)

    input_dim = X_train_sequences.shape[2]

    if input_dim % num_heads != 0:
        num_heads = 1
        wandb.config.num_heads = num_heads

    model = TransformerModel(
        input_dim=input_dim, 
        num_heads=num_heads, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_classes=output_dim, 
        dropout=dropout, 
        activation=activation,
        loss_type=loss_type
    )

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    if loss_type == "binary_crossentropy":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "categorical_crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_train_sequences), torch.Tensor(y_train_sequences).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_val_sequences), torch.Tensor(y_val_sequences).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_test_sequences), torch.Tensor(y_test_sequences).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False
    )

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        y_true_train, y_pred_train = [], []

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            y_true_train.extend(target.cpu().numpy())
            y_pred_train.extend(output.detach().cpu().numpy())

        train_accuracy = accuracy_score(y_true_train, np.round(y_pred_train))
        train_precision = precision_score(y_true_train, np.round(y_pred_train))
        train_recall = recall_score(y_true_train, np.round(y_pred_train))
        train_f1 = f1_score(y_true_train, np.round(y_pred_train))

        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, '
              f'Train Accuracy: {train_accuracy}, Precision: {train_precision}, '
              f'Recall: {train_recall}, F1-score: {train_f1}')

        wandb.log({
                    'train_loss': train_loss / len(train_loader),
                    'train_accuracy': train_accuracy,
                    'train_precision': train_precision,
                    'train_recall': train_recall,
                    'train_f1': train_f1,
                    'epoch': epoch + 1
                })

        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                y_true_val.extend(target.cpu().numpy())
                y_pred_val.extend(output.cpu().numpy())

        val_accuracy = accuracy_score(y_true_val, np.round(y_pred_val))
        val_precision = precision_score(y_true_val, np.round(y_pred_val))
        val_recall = recall_score(y_true_val, np.round(y_pred_val))
        val_f1 = f1_score(y_true_val, np.round(y_pred_val))

        print(f'Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader)}, '
              f'Val Accuracy: {val_accuracy}, Precision: {val_precision}, '
              f'Recall: {val_recall}, F1-score: {val_f1}')
        
        wandb.log({
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'epoch': epoch + 1
        })

    model.eval()
    test_loss = 0.0
    y_true_test, y_pred_test = [], []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            y_true_test.extend(target.cpu().numpy())
            y_pred_test.extend(output.cpu().numpy())

    test_accuracy = accuracy_score(y_true_test, np.round(y_pred_test))
    metrics_test = get_metrics(np.round(y_pred_test), y_true_test)
    wandb.log({
        'test_loss': test_loss / len(test_loader),
        'test_accuracy': test_accuracy,
        **metrics_test
    })
    print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}, Test Metrics: {metrics_test}')


def main():
    df = pd.read_csv("preprocessing/merged_features/all_participants_normalized.csv")

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    sweep_config = {
        'method': 'random',
        'name': 'transformer_sweep',
        'parameters': {
            'num_heads': {'values': [1, 7, 25, 100]},
            'num_layers': {'values': [2, 3]},
            'hidden_dim': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid', 'softmax']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'RMSprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'epochs': {'value': 500},
            'loss': {'values': ["binary_crossentropy", "categorical_crossentropy"]},
            'sequence_length': {'values': [1, 5, 15, 30]}
        }
    }

    def train_wrapper():
        train(df)

    sweep_id = wandb.sweep(sweep=sweep_config, project="transformer_sweep_v1")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
