import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import wandb
from create_data_splits import create_data_splits, create_data_splits_pca

def get_metrics(preds, targets):
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='binary')
    recall = recall_score(targets, preds, average='binary')
    f1 = f1_score(targets, preds, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
    
def transformer_model(df, config):

    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_heads = config.num_heads
    hidden_dim = config.hidden_dim
    num_layers = config.num_layers
    batch_size = config.batch_size
    activation = config.activation_function
    dropout = config.dropout_rate
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    loss = config.loss
    sequence_length = config.sequence_length
    output_dim = 1
    epochs = config.epochs
    data = config.data

    for fold in range(5):

        splits = None

        if data == "reg" or data == "norm":
            splits = create_data_splits(
                df, "binary",
                fold_no=5,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
            if splits is None:
                return

        elif data == "pca":
            splits = create_data_splits_pca(
                df, "binary",
                fold_no=5,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
            if splits is None:
                return

        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

        print("X_train_sequences shape:", X_train_sequences.shape)
        print("X_val_sequences shape:", X_val_sequences.shape)
        print("X_test_sequences shape:", X_test_sequences.shape)

        input_dim = X_train_sequences.shape[2]
        output_dim = 1

        if input_dim % num_heads != 0:
            num_heads = 1
            config.num_heads = 1
        
        model = TransformerModel(
            input_dim=input_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            num_classes=1, 
            dropout=dropout, 
            activation=activation, 
            loss_type=loss
        ).to(device)

        optimizer = {
            'adam': torch.optim.Adam(model.parameters(), lr=learning_rate),
            'sgd': torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
            'adadelta': torch.optim.Adadelta(model.parameters(), lr=learning_rate),
            'RMSprop': torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        }.get(optimizer, torch.optim.Adam(model.parameters(), lr=learning_rate))

        criterion = {
            'binary_crossentropy': nn.BCEWithLogitsLoss(),
            'categorical_crossentropy': nn.CrossEntropyLoss()
        }.get(config.loss, nn.BCEWithLogitsLoss())

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_train_sequences), 
                torch.Tensor(y_train_sequences).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_val_sequences), 
                torch.Tensor(y_val_sequences).unsqueeze(1)
            ),
            batch_size=config.batch_size,
            shuffle=False
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_test_sequences), 
                torch.Tensor(y_test_sequences).unsqueeze(1)
            ),
            batch_size=config.batch_size,
            shuffle=False
        )
        
        def run_epoch(loader, model, criterion, optimizer=None, train=True):
            epoch_loss = 0.0
            y_true, y_pred = [], []
            
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                if train:
                    model.train()
                    optimizer.zero_grad()
                else:
                    model.eval()

                with torch.set_grad_enabled(train):
                    output = model(data)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    if train:
                        loss.backward()
                        optimizer.step()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(torch.sigmoid(output).detach().cpu().numpy())

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            y_pred = (y_pred > 0.5).astype(int)

            metrics = get_metrics(y_pred, y_true)
            return epoch_loss / len(loader), metrics

        for epoch in range(epochs):
            train_loss, train_metrics = run_epoch(train_loader, model, criterion, optimizer, train=True)
            val_loss, val_metrics = run_epoch(val_loader, model, criterion, train=False)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, '
                f'Train Accuracy: {train_metrics["accuracy"]:.4f}, Precision: {train_metrics["precision"]:.4f}, '
                f'Recall: {train_metrics["recall"]:.4f}, F1-score: {train_metrics["f1"]:.4f}')

            wandb.log({
                'Train Loss': train_loss,
                'Train Accuracy': train_metrics['accuracy'],
                'Train Precision': train_metrics['precision'],
                'Train Recall': train_metrics['recall'],
                'Train F1-score': train_metrics['f1'],
                'Epoch': epoch + 1
            })

            print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, '
                f'Val Accuracy: {val_metrics["accuracy"]:.4f}, Precision: {val_metrics["precision"]:.4f}, '
                f'Recall: {val_metrics["recall"]:.4f}, F1-score: {val_metrics["f1"]:.4f}')
            
            wandb.log({
                'Val Loss': val_loss,
                'Val Accuracy': val_metrics['accuracy'],
                'Val Precision': val_metrics['precision'],
                'Val Recall': val_metrics['recall'],
                'Val F1-score': val_metrics['f1'],
                'Epoch': epoch + 1
            })

        test_loss, test_metrics = run_epoch(test_loader, model, criterion, train=False)
        wandb.log({
            'Test Loss': test_loss,
            'Test Accuracy': test_metrics['accuracy'],
            'Test Precision': test_metrics['precision'],
            'Test Recall': test_metrics['recall'],
            'Test F1-score': test_metrics['f1'],
            'Epoch': epoch + 1
        })

        print(f'Test Loss: {test_loss:.4f}, '
            f'Test Accuracy: {test_metrics["accuracy"]:.4f}, Precision: {test_metrics["precision"]:.4f}, '
            f'Recall: {test_metrics["recall"]:.4f}, F1-score: {test_metrics["f1"]:.4f}')

        wandb.finish()

def train():

    wandb.init()
    config = wandb.config
    print(config)

    modality = config.modality
    feature_set = config.feature_set
    data = config.data

    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")

    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
    df_audio_index = df.iloc[:, 63:88]

    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]

    df_facial_index_rf = df_rf.iloc[:, 38:]
    df_pose_index_rf = df_rf.iloc[:, 4:28]
    df_audio_index_rf = df_rf.iloc[:, 28:38]

    modality_mapping = {
        "pose": pd.concat([info, df_pose_index], axis=1),
        "facial": pd.concat([info, df_facial_index], axis=1),
        "audio": pd.concat([info, df_audio_index], axis=1)
    }

    modality_mapping_stats = {
        "facial": pd.concat([info, df_facial_index_stats], axis=1),
        "audio": pd.concat([info, df_audio_index_stats], axis=1)
    }

    modality_mapping_rf = {
        "pose": pd.concat([info, df_pose_index_rf], axis=1),
        "facial": pd.concat([info, df_facial_index_rf], axis=1),
        "audio": pd.concat([info, df_audio_index_rf], axis=1)
    }

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

    def get_modality_data(modality, data):
        df = pd.DataFrame()

        if feature_set == "full":
            df = modality_mapping.get(modality)

        elif feature_set == "stats":
            df = modality_mapping_stats.get(modality)

        elif feature_set == "rf":
            df = modality_mapping_rf.get(modality)

        if data != "reg":
            df = create_normalized_df(df)

        return df

    transformer_model(get_modality_data(modality, data), config)


def main():
    
    modality = "audio"

    sweep_config = {
        'method': 'random',
        'name': f'transformer_{modality}',
        'parameters': {
            'modality' : {'value': modality},
            
            'feature_set' : {'values': ["full", "stats", "rf"]},
            'data' : {'values' : ["reg", "norm", "pca"]},

            'num_heads': {'values': [2, 4, 8, 16, 32, 100]},
            'num_layers': {'values': [2, 3]},
            'hidden_dim': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid', 'softmax']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'RMSprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'epochs': {'value': 500},
            'loss': {'values': ["binary_crossentropy", "categorical_crossentropy"]},
            'sequence_length': {'values': [1, 5, 15, 30, 60, 90]},
        }
    }

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project=f"transformer_{modality}")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
