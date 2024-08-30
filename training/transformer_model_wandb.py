import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb

from create_data_splits import create_data_splits, create_data_splits_pca
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

class PoseTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(PoseTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True),
            num_layers=6
        )
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class FacialTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FacialTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True),
            num_layers=6
        )
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class AudioTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(AudioTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True),
            num_layers=6
        )
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class IntermediateFusionModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, num_classes, dropout, activation, loss_type):
        super(IntermediateFusionModel, self).__init__()

        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.facial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        
        self.fc_pose = nn.Linear(input_dim, hidden_dim)
        self.fc_facial = nn.Linear(input_dim, hidden_dim)
        self.fc_audio = nn.Linear(input_dim, hidden_dim)

        self.fc_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, num_classes)

        if loss_type == "binary_crossentropy":
            self.output_activation = nn.Sigmoid()
        elif loss_type == "categorical_crossentropy":
            self.output_activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
    def forward(self, pose_input, facial_input, audio_input):
        pose_features = self.pose_transformer(pose_input)
        pose_features = pose_features.mean(dim=1)
        pose_features = self.fc_pose(pose_features)
        
        facial_features = self.facial_transformer(facial_input)
        facial_features = facial_features.mean(dim=1)
        facial_features = self.fc_facial(facial_features)
        
        audio_features = self.audio_transformer(audio_input)
        audio_features = audio_features.mean(dim=1)
        audio_features = self.fc_audio(audio_features)
        
        combined_features = torch.cat((pose_features, facial_features, audio_features), dim=1)
        fused_features = torch.relu(self.fc_fusion(combined_features))
        
        output = self.fc_output(fused_features)
        output = self.output_activation(output)
        return output

class LateFusionModel(nn.Module):
    def __init__(self, pose_dim, facial_dim, audio_dim, num_heads, hidden_dim, num_layers, num_classes, dropout, activation, loss_type):
        super(LateFusionModel, self).__init__()
        
        self.pose_transformer = TransformerModel(input_dim=pose_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        self.facial_transformer = TransformerModel(input_dim=facial_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        self.audio_transformer = TransformerModel(input_dim=audio_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, pose_input, facial_input, audio_input):
        pose_features = self.pose_transformer(pose_input)
        facial_features = self.facial_transformer(facial_input)
        audio_features = self.audio_transformer(audio_input)
        
        combined_features = torch.cat((pose_features, facial_features, audio_features), dim=1)
        
        x = torch.relu(self.fc1(combined_features))
        output = self.fc2(x)
        
        return output


def train(df, config):

    wandb.init(config=config)
    print(config)

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
    use_pca = config.use_pca
    epochs = config.epochs
    fusion_type = config.fusion_type
    
    
    if fusion_type == "early":

        if use_pca:
            splits = create_data_splits_pca(df, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        else:
            splits = create_data_splits(df, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        
        if splits is None:
            return

        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

        print("X_train_sequences shape:", X_train_sequences.shape)
        print("X_val_sequences shape:", X_val_sequences.shape)
        print("X_test_sequences shape:", X_test_sequences.shape)

        input_dim = X_train_sequences.shape[2]
        output_dim = 1
        
        if num_heads and input_dim % num_heads != 0:
            config.num_heads = 1
        
        model = TransformerModel(
            input_dim=input_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            num_classes=2, 
            dropout=dropout, 
            activation=activation, 
            loss_type=loss
        )

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
                y_pred.extend(output.detach().cpu().numpy())

            metrics = get_metrics(np.round(y_pred), y_true)
            return epoch_loss / len(loader), metrics

        for epoch in range(epochs):
            train_loss, train_metrics = run_epoch(train_loader, model, criterion, optimizer, train=True)
            val_loss, val_metrics = run_epoch(val_loader, model, criterion, train=False)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, '
                f'Train Accuracy: {train_metrics["accuracy"]}, Precision: {train_metrics["precision"]}, '
                f'Recall: {train_metrics["recall"]}, F1-score: {train_metrics["f1"]}')

            wandb.log({**train_metrics, 'train_loss': train_loss, 'epoch': epoch + 1})

            print(f'Epoch {epoch+1}, Val Loss: {val_loss}, '
                f'Val Accuracy: {val_metrics["accuracy"]}, Precision: {val_metrics["precision"]}, '
                f'Recall: {val_metrics["recall"]}, F1-score: {val_metrics["f1"]}')
            
            wandb.log({**val_metrics, 'val_loss': val_loss, 'epoch': epoch + 1})

        test_loss, test_metrics = run_epoch(test_loader, model, criterion, train=False)
        wandb.log({**test_metrics, 'test_loss': test_loss, 'epoch': epoch + 1})

        print(f'Test Loss: {test_loss}, '
            f'Test Accuracy: {test_metrics["accuracy"]}, Precision: {test_metrics["precision"]}, '
            f'Recall: {test_metrics["recall"]}, F1-score: {test_metrics["f1"]}')

        wandb.finish()

    elif fusion_type == "intermediate":

        df_pose = df.iloc[:, 4:29]
        df_facial = df.iloc[:, 29:65]
        df_audio = df.iloc[:, 65:]

        if use_pca:
            splits_pose = create_data_splits_pca(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits_pca(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits_pca(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        else:
            splits_pose = create_data_splits(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_pose is None or splits_facial is None or splits_audio is None:
            return
        
        X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
        X_train_facial, _, _, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
        X_train_audio, _, _, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

        model = IntermediateFusionModel(
            input_dim=input_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            num_classes=2, 
            dropout=dropout, 
            activation=activation, 
            loss_type=loss
        )

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
            torch.Tensor(X_train_pose_seq), 
            torch.Tensor(X_train_facial_seq),
            torch.Tensor(X_train_audio_seq),
            torch.Tensor(y_train_sequences).unsqueeze(1)
        ),
        batch_size=batch_size,
        shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_val_pose_seq), 
                torch.Tensor(X_val_facial_seq),
                torch.Tensor(X_val_audio_seq),
                torch.Tensor(y_val_sequences).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_test_pose_seq), 
                torch.Tensor(X_test_facial_seq),
                torch.Tensor(X_test_audio_seq),
                torch.Tensor(y_test_sequences).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        def run_epoch(loader, model, criterion, optimizer=None, train=True):
            epoch_loss = 0.0
            y_true, y_pred = [], []
            
            for pose_input, facial_input, audio_input, target in loader:
                if train:
                    model.train()
                    optimizer.zero_grad()
                else:
                    model.eval()

                with torch.set_grad_enabled(train):
                    output = model(pose_input, facial_input, audio_input)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    if train:
                        loss.backward()
                        optimizer.step()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(output.detach().cpu().numpy())

            metrics = get_metrics(np.round(y_pred), y_true)
            return epoch_loss / len(loader), metrics

        for epoch in range(epochs):
            train_loss, train_metrics = run_epoch(train_loader, model, criterion, optimizer, train=True)
            val_loss, val_metrics = run_epoch(val_loader, model, criterion, train=False)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, '
                f'Train Accuracy: {train_metrics["accuracy"]}, Precision: {train_metrics["precision"]}, '
                f'Recall: {train_metrics["recall"]}, F1-score: {train_metrics["f1"]}')

            wandb.log({**train_metrics, 'train_loss': train_loss, 'epoch': epoch + 1})

            print(f'Epoch {epoch+1}, Val Loss: {val_loss}, '
                f'Val Accuracy: {val_metrics["accuracy"]}, Precision: {val_metrics["precision"]}, '
                f'Recall: {val_metrics["recall"]}, F1-score: {val_metrics["f1"]}')
            
            wandb.log({**val_metrics, 'val_loss': val_loss, 'epoch': epoch + 1})

        test_loss, test_metrics = run_epoch(test_loader, model, criterion, train=False)
        wandb.log({**test_metrics, 'test_loss': test_loss, 'epoch': epoch + 1})

        print(f'Test Loss: {test_loss}, '
            f'Test Accuracy: {test_metrics["accuracy"]}, Precision: {test_metrics["precision"]}, '
            f'Recall: {test_metrics["recall"]}, F1-score: {test_metrics["f1"]}')

        wandb.finish()


    elif fusion_type == "late":

        df_pose = df.iloc[:, 4:29]
        df_facial = df.iloc[:, 29:65]
        df_audio = df.iloc[:, 65:]

        if use_pca:
            splits_pose = create_data_splits_pca(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits_pca(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits_pca(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        else:
            splits_pose = create_data_splits(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_pose is None or splits_facial is None or splits_audio is None:
            return
        
        X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
        X_train_facial, _, _, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
        X_train_audio, _, _, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

        model = LateFusionModel(
            input_dim=input_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            num_classes=2, 
            dropout=dropout, 
            activation=activation, 
            loss_type=loss
        )

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
            torch.Tensor(X_train_pose_seq), 
            torch.Tensor(X_train_facial_seq),
            torch.Tensor(X_train_audio_seq),
            torch.Tensor(y_train_sequences).unsqueeze(1)
        ),
        batch_size=batch_size,
        shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_val_pose_seq), 
                torch.Tensor(X_val_facial_seq),
                torch.Tensor(X_val_audio_seq),
                torch.Tensor(y_val_sequences).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(X_test_pose_seq), 
                torch.Tensor(X_test_facial_seq),
                torch.Tensor(X_test_audio_seq),
                torch.Tensor(y_test_sequences).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        def run_epoch(loader, model, criterion, optimizer=None, train=True):
            epoch_loss = 0.0
            y_true, y_pred = [], []
            
            for pose_input, facial_input, audio_input, target in loader:
                if train:
                    model.train()
                    optimizer.zero_grad()
                else:
                    model.eval()

                with torch.set_grad_enabled(train):
                    output = model(pose_input, facial_input, audio_input)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    if train:
                        loss.backward()
                        optimizer.step()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(output.detach().cpu().numpy())

            metrics = get_metrics(np.round(y_pred), y_true)
            return epoch_loss / len(loader), metrics

        for epoch in range(epochs):
            train_loss, train_metrics = run_epoch(train_loader, model, criterion, optimizer, train=True)
            val_loss, val_metrics = run_epoch(val_loader, model, criterion, train=False)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, '
                f'Train Accuracy: {train_metrics["accuracy"]}, Precision: {train_metrics["precision"]}, '
                f'Recall: {train_metrics["recall"]}, F1-score: {train_metrics["f1"]}')

            wandb.log({**train_metrics, 'train_loss': train_loss, 'epoch': epoch + 1})

            print(f'Epoch {epoch+1}, Val Loss: {val_loss}, '
                f'Val Accuracy: {val_metrics["accuracy"]}, Precision: {val_metrics["precision"]}, '
                f'Recall: {val_metrics["recall"]}, F1-score: {val_metrics["f1"]}')
            
            wandb.log({**val_metrics, 'val_loss': val_loss, 'epoch': epoch + 1})

        test_loss, test_metrics = run_epoch(test_loader, model, criterion, train=False)
        wandb.log({**test_metrics, 'test_loss': test_loss, 'epoch': epoch + 1})

        print(f'Test Loss: {test_loss}, '
            f'Test Accuracy: {test_metrics["accuracy"]}, Precision: {test_metrics["precision"]}, '
            f'Recall: {test_metrics["recall"]}, F1-score: {test_metrics["f1"]}')

        wandb.finish()



def main():

    df = pd.read_csv("all_participants_merged_correct_normalized.csv")

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    sweep_config = {
        'method': 'random',
        'name': 'transformer_sweep',
        'parameters': {
            'use_pca': {'values': [True, False]},
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
            'sequence_length': {'values': [1, 5, 15, 30, 60, 90]},
            'fusion_type' : {'values' : ['early', 'intermediate', 'late']}
        }
    }

    def train_wrapper():
        train(df)

    sweep_id = wandb.sweep(sweep=sweep_config, project="transformer_sweep_v1")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
