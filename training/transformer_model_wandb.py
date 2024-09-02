import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

class IntermediateFusionModel(nn.Module):
    def __init__(self, pose_dim, facial_dim, audio_dim, num_heads, hidden_dim, num_layers, num_classes, dropout, activation, loss_type):
        super(IntermediateFusionModel, self).__init__()

        self.pose_transformer = TransformerModel(input_dim=pose_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        self.facial_transformer = TransformerModel(input_dim=facial_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        self.audio_transformer = TransformerModel(input_dim=audio_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)

        self.fc_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, pose_input, facial_input, audio_input):
        pose_features = self.pose_transformer(pose_input)
        facial_features = self.facial_transformer(facial_input)
        audio_features = self.audio_transformer(audio_input)

        # print("pose " + pose_features.shape)
        # print("face " + facial_features.shape)
        # print("audio " + audio_features.shape)
        
        combined_features = torch.cat((pose_features, facial_features, audio_features), dim=-1)
        combined_features = self.dropout(combined_features)

        # print("combined " + combined_features.shape)

        fused_features = self.fc_fusion(combined_features)
        fused_features = self.dropout(fused_features)

        # print("fusion " + fused_features.shape)

        fused_features = torch.relu(fused_features)
        fused_features = self.dropout(fused_features)
        
        output = self.fc_output(fused_features)
        
        return output

class LateFusionModel(nn.Module):
    def __init__(self, pose_dim, facial_dim, audio_dim, num_heads, hidden_dim, num_layers, num_classes, dropout, activation, loss_type):
        super(LateFusionModel, self).__init__()
        
        self.pose_transformer = TransformerModel(input_dim=pose_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        self.facial_transformer = TransformerModel(input_dim=facial_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)
        self.audio_transformer = TransformerModel(input_dim=audio_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=hidden_dim, dropout=dropout, activation=activation, loss_type=loss_type)

        self.fc_fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, pose_input, facial_input, audio_input):
        pose_features = self.pose_transformer(pose_input)
        facial_features = self.facial_transformer(facial_input)
        audio_features = self.audio_transformer(audio_input)

        # print("pose " , pose_features.shape)
        # print("face " , facial_features.shape)
        # print("audio " , audio_features.shape)
        
        combined_features = torch.cat((pose_features, facial_features, audio_features), dim=1)
        combined_features = self.dropout(combined_features)

        # print("combined " , combined_features.shape)

        fused_features = self.fc_fusion(combined_features)
        fused_features = self.dropout(fused_features)

        # print("fusion " , fused_features.shape)

        output = self.output_layer(fused_features)
        
        return output


def train(df, config):

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

    elif fusion_type == "intermediate":

        participant_frames_labels = df.iloc[:, :4]

        df_pose = df.iloc[:, 4:29]
        df_pose = pd.concat([participant_frames_labels, df_pose], axis=1)
        df_facial = df.iloc[:, 29:65]
        df_facial = pd.concat([participant_frames_labels, df_facial], axis=1)
        df_audio = df.iloc[:, 65:]
        df_audio = pd.concat([participant_frames_labels, df_audio], axis=1)

        if use_pca:
            print("pose split")
            splits_pose = create_data_splits_pca(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("face split")
            splits_facial = create_data_splits_pca(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("audio split")
            splits_audio = create_data_splits_pca(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        else:
            print("pose split")
            splits_pose = create_data_splits(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("face split")
            splits_facial = create_data_splits(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("audio split")
            splits_audio = create_data_splits(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_pose is None or splits_facial is None or splits_audio is None:
            return
        
        X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
        X_train_facial, _, _, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
        X_train_audio, _, _, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

        pose_dim, facial_dim, audio_dim = X_train_pose.shape[-1], X_train_facial.shape[-1], X_train_audio.shape[-1]

        if pose_dim % num_heads != 0 or facial_dim % num_heads != 0 or audio_dim % num_heads != 0:
            num_heads = 1
            config.num_heads = 1

        model = IntermediateFusionModel(
            pose_dim=pose_dim,
            facial_dim=facial_dim,
            audio_dim=audio_dim,
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
            
            for batch in loader:
                print(batch)

                pose_data, facial_data, audio_data, target = batch

                pose_data, facial_data, audio_data, target = (
                    pose_data.to(device),
                    facial_data.to(device),
                    audio_data.to(device),
                    target.to(device)
                )
                
                if train:
                    model.train()
                    optimizer.zero_grad()
                else:
                    model.eval()

                with torch.set_grad_enabled(train):
                    output = model(pose_data, facial_data, audio_data)
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

    elif fusion_type == "late":

        participant_frames_labels = df.iloc[:, :4]
        print(participant_frames_labels)

        df_pose = df.iloc[:, 4:29]
        df_pose = pd.concat([participant_frames_labels, df_pose], axis=1)
        df_facial = df.iloc[:, 29:65]
        df_facial = pd.concat([participant_frames_labels, df_facial], axis=1)
        df_audio = df.iloc[:, 65:]
        df_audio = pd.concat([participant_frames_labels, df_audio], axis=1)

        if use_pca:
            print("pose split")
            splits_pose = create_data_splits_pca(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("face split")
            splits_facial = create_data_splits_pca(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("audio split")
            splits_audio = create_data_splits_pca(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        else:
            print("pose split")
            splits_pose = create_data_splits(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("face split")
            splits_facial = create_data_splits(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            print("audio split")
            splits_audio = create_data_splits(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_pose is None or splits_facial is None or splits_audio is None:
            return
        
        X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
        X_train_facial, _, _, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
        X_train_audio, _, _, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

        pose_dim, facial_dim, audio_dim = X_train_pose.shape[-1], X_train_facial.shape[-1], X_train_audio.shape[-1]

        if pose_dim % num_heads != 0 or facial_dim % num_heads != 0 or audio_dim % num_heads != 0:
            num_heads = 1
            config.num_heads = 1

        model = LateFusionModel(
            pose_dim=pose_dim,
            facial_dim=facial_dim,
            audio_dim=audio_dim,
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
            
            for batch in loader:
                # print(batch)

                pose_data, facial_data, audio_data, target = batch

                pose_data, facial_data, audio_data, target = (
                    pose_data.to(device),
                    facial_data.to(device),
                    audio_data.to(device),
                    target.to(device)
                )

                if train:
                    model.train()
                    optimizer.zero_grad()
                else:
                    model.eval()

                with torch.set_grad_enabled(train):
                    output = model(pose_data, facial_data, audio_data)
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


def main():

    df = pd.read_csv("preprocessing/merged_features/all_participants_merged_correct_normalized.csv")

    sweep_config = {
        'method': 'random',
        'name': 'transformer_sweep',
        'parameters': {
            'use_pca': {'values': [True, False]},
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
            'fusion_type' : {'values' : ['early', 'intermediate', 'late']}
        }
    }

    def train_wrapper():
        wandb.init()
        config = wandb.config
        train(df, config)

    sweep_id = wandb.sweep(sweep=sweep_config, project="transformer_sweep_v1")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
