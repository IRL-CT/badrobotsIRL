import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2, l1, l2
from keras.utils import to_categorical
import tensorflow as tf
from create_data_splits import create_data_splits, create_data_splits_pca
from get_metrics import get_test_metrics

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

        self.loss_type = loss_type
        if loss_type == "binary_crossentropy":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None
        
        self.activation_function = activation

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)

        if self.activation_function:
            x = self.activation_function(x)

        x = self.fc(x)

        if self.loss_type == "binary_crossentropy":
            x = torch.sigmoid(x)

        return x

def transformer_model(feature_name, X_train_sequences, X_val_sequences, y_train, y_train_sequences, X_test_sequences, y_val, y_test, y_val_sequences, config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_train_sequences.shape[2]
    num_classes = len(np.unique(y_train_sequences))

    num_heads = max(2, config.transformer_num_heads // 2 * 2)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_sequences)):
        print(f"Training fold {fold + 1}/5...")

        X_train, X_val = X_train_sequences[train_idx], X_val_sequences[val_idx]
        y_train, y_val = y_train_sequences[train_idx], y_val_sequences[val_idx]
        
        model = TransformerModel(
            input_dim=input_dim, 
            num_heads=num_heads, 
            hidden_dim=config.transformer_hidden_dim, 
            num_layers=config.transformer_num_layers, 
            num_classes=num_classes, 
            dropout=config.dropout, 
            activation=config.activation_function, 
            loss_type=config.loss
        ).to(device)

        if config.loss == "categorical_crossentropy":
            criterion = nn.CrossEntropyLoss()
        elif config.loss == "binary_crossentropy":
            criterion = nn.BCEWithLogitsLoss() 
        else:
            raise ValueError("Unsupported loss function")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if config.loss == "binary_crossentropy" else torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32 if config.loss == "binary_crossentropy" else torch.long).to(device)

        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)

            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            metrics = {
                'epoch': epoch,
                'feature': feature_name,
                'fold': fold,
                'train_loss': loss.item(),
                'val_loss': val_loss.item()
            }
            wandb.log(metrics)
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_val_tensor).cpu().numpy()

        fold_metrics.append({
            'feature': feature_name,
            "fold": fold,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "predictions": predictions
        })
    
    return predictions

def lstm_model(feature_name, X_train_sequences, X_val_sequences, y_train, y_train_sequences, X_test_sequences, y_val, y_test, y_val_sequences, config):
    
    input_shape = X_train_sequences.shape[2]

    model = Sequential()
    model.add(Input(shape=(config.sequence_length, input_shape)))

    if config.lstm_num_layers == 1:
        if config.lstm_use_bidirectional:
            model.add(Bidirectional(GRU(config.lstm_units, activation=config.activation_function, kernel_regularizer=config.reg)))
        else:
            model.add(LSTM(config.lstm_units, activation=config.activation_function, kernel_regularizer=config.reg))
        model.add(Dropout(config.dropout))
        model.add(BatchNormalization())
    else:
        for _ in range(config.lstm_num_layers - 1):
            if config.lstm_use_bidirectional:
                model.add(Bidirectional(LSTM(config.lstm_units, return_sequences=True, activation=config.activation_function, kernel_regularizer=config.reg)))
            else:
                model.add(LSTM(config.gru_units, return_sequences=True, activation=config.activation_function, kernel_regularizer=config.reg))
            model.add(Dropout(config.dropout))
            model.add(BatchNormalization())

        if config.lstm_use_bidirectional:
            model.add(Bidirectional(LSTM(config.lstm_units, activation=config.activation_function)))
        else:
            model.add(LSTM(config.lstm_units, activation=config.activation_function))
        model.add(Dropout(config.dropout))
        model.add(BatchNormalization())
    
    num_classes = len(np.unique(y_train))
    print("Num classes: ", num_classes)
    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_val:", np.unique(y_val))
    print("Unique labels in y_test:", np.unique(y_test))

    model.add(Dense(config.dense_units, activation=config.activation_function))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()
    
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    
    model.fit(X_train_sequences, y_train_sequences, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_val_sequences, y_val_sequences), verbose=2)
    
    for epoch in range(len(model_history.history['loss'])):

        metrics = {
            'fold': fold,
            'feature': feature_name,
            'epoch': epoch,
            'loss': model_history.history['loss'][epoch],
            'val_loss': model_history.history['val_loss'][epoch]
        }

        if 'loss' in model_history.history:
            metrics['total_train_loss'] = model_history.history['loss'][epoch]
        if 'val_loss' in model_history.history:
            metrics['total_val_loss'] = model_history.history['val_loss'][epoch]

        if 'accuracy' in model_history.history:
            metrics['train_accuracy'] = model_history.history['accuracy'][epoch]
        if 'val_accuracy' in model_history.history:
            metrics['val_accuracy'] = model_history.history['val_accuracy'][epoch]
        
        if 'precision' in model_history.history:
            metrics['train_precision'] = model_history.history['precision'][epoch]
        if 'val_precision' in model_history.history:
            metrics['val_precision'] = model_history.history['val_precision'][epoch]
        
        if 'recall' in model_history.history:
            metrics['train_recall'] = model_history.history['recall'][epoch]
        if 'val_recall' in model_history.history:
            metrics['val_recall'] = model_history.history['val_recall'][epoch]
        
        if 'auc' in model_history.history:
            metrics['train_auc'] = model_history.history['auc'][epoch]
        if 'val_auc' in model_history.history:
            metrics['val_auc'] = model_history.history['val_auc'][epoch]

        wandb.log(metrics)

    return model.predict(X_test_sequences)


def gru_model(feature_name, X_train_sequences, X_val_sequences, y_train, y_train_sequences, X_test_sequences, y_val, y_test, y_val_sequences, config):

    input_shape = X_train_sequences.shape[2]

    model = Sequential()
    model.add(Input(shape=(config.sequence_length, input_shape)))

    if config.gru_num_layers == 1:
        if config.gru_use_bidirectional:
            model.add(Bidirectional(GRU(config.gru_units, activation=config.activation_function, kernel_regularizer=config.reg)))
        else:
            model.add(GRU(config.gru_units, activation=config.activation_function, kernel_regularizer=config.reg))
        model.add(Dropout(config.dropout))
        model.add(BatchNormalization())
    else:
        for _ in range(config.gru_num_layers - 1):
            if config.gru_use_bidirectional:
                model.add(Bidirectional(GRU(config.gru_units, return_sequences=True, activation=config.activation_function, kernel_regularizer=config.reg)))
            else:
                model.add(GRU(config.gru_units, return_sequences=True, activation=config.activation_function, kernel_regularizer=config.reg))
            model.add(Dropout(config.dropout))
            model.add(BatchNormalization())

        if config.gru_use_bidirectional:
            model.add(Bidirectional(GRU(config.gru_units, activation=config.activation_function)))
        else:
            model.add(GRU(config.gru_units, activation=config.activation_function))
        model.add(Dropout(config.dropout))
        model.add(BatchNormalization())
    
    num_classes = len(np.unique(y_train))
    print("Num classes: ", num_classes)
    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_val:", np.unique(y_val))
    print("Unique labels in y_test:", np.unique(y_test))

    model.add(Dense(config.dense_units, activation=config.activation_function))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()
    
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    
    model.fit(X_train_sequences, y_train_sequences, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_val_sequences, y_val_sequences), verbose=2)
    
    for epoch in range(len(model_history.history['loss'])):

        metrics = {
            'fold': fold,
            'feature': feature_name,
            'epoch': epoch,
            'loss': model_history.history['loss'][epoch],
            'val_loss': model_history.history['val_loss'][epoch]
        }

        if 'loss' in model_history.history:
            metrics['total_train_loss'] = model_history.history['loss'][epoch]
        if 'val_loss' in model_history.history:
            metrics['total_val_loss'] = model_history.history['val_loss'][epoch]

        if 'accuracy' in model_history.history:
            metrics['train_accuracy'] = model_history.history['accuracy'][epoch]
        if 'val_accuracy' in model_history.history:
            metrics['val_accuracy'] = model_history.history['val_accuracy'][epoch]
        
        if 'precision' in model_history.history:
            metrics['train_precision'] = model_history.history['precision'][epoch]
        if 'val_precision' in model_history.history:
            metrics['val_precision'] = model_history.history['val_precision'][epoch]
        
        if 'recall' in model_history.history:
            metrics['train_recall'] = model_history.history['recall'][epoch]
        if 'val_recall' in model_history.history:
            metrics['val_recall'] = model_history.history['val_recall'][epoch]
        
        if 'auc' in model_history.history:
            metrics['train_auc'] = model_history.history['auc'][epoch]
        if 'val_auc' in model_history.history:
            metrics['val_auc'] = model_history.history['val_auc'][epoch]

        wandb.log(metrics)
    
    return model.predict(X_test_sequences)


def ast_model(X_train_sequences, X_val_sequences, y_train, y_train_sequences, X_test_sequences, y_val, y_test, y_val_sequences, config):
    return


def train():
    wandb.init()
    config = wandb.config
    print(config)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")

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
        print("Fold ", fold)
    
        info = df.iloc[:, :4]
        df_pose_index = df.iloc[:, 4:28]
        df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
        df_audio_index = df.iloc[:, 63:88]

        df_pose = pd.concat([info, df_pose_index], axis=1)
        df_facial = pd.concat([info, df_facial_index], axis=1)
        df_audio = pd.concat([info, df_audio_index], axis=1)

        splits = create_data_splits(df, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=config.sequence_length)
        if splits is None:
            return
        
        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

        X_train_pose = X_train.iloc[:, 4:28]
        X_train_facial = np.concatenate([X_train.iloc[:, 28:63], X_train.iloc[:, 88:]], axis=1)
        X_train_audio = X_train.iloc[:, 63:88]

        X_val_pose = X_val.iloc[:, 4:28]
        X_val_facial = np.concatenate([X_val.iloc[:, 28:63], X_val.iloc[:, 88:]], axis=1)
        X_val_audio = X_val.iloc[:, 63:88]

        X_test_pose = X_test.iloc[:, 4:28]
        X_test_facial = np.concatenate([X_test.iloc[:, 28:63], X_test.iloc[:, 88:]], axis=1)
        X_test_audio = X_test.iloc[:, 63:88]

        X_train_sequences_pose = X_train_sequences[:, :, 4:28]
        X_train_sequences_facial = np.concatenate([X_train_sequences[:, :, 28:63], X_train_sequences[:, :, 88:]], axis=2)
        X_train_sequences_audio = X_train_sequences[:, :, 63:88]

        X_val_sequences_pose = X_val_sequences[:, :, 4:28]
        X_val_sequences_facial = np.concatenate([X_val_sequences[:, :, 28:63], X_val_sequences[:, :, 88:]], axis=2)
        X_val_sequences_audio = X_val_sequences[:, :, 63:88]

        X_test_sequences_pose = X_test_sequences[:, :, 4:28]
        X_test_sequences_facial = np.concatenate([X_test_sequences[:, :, 28:63], X_test_sequences[:, :, 88:]], axis=2)
        X_test_sequences_audio = X_test_sequences[:, :, 63:88]

        y_train_sequences = to_categorical(y_train_sequences, num_classes=4)
        y_val_sequences = to_categorical(y_val_sequences, num_classes=4)
        y_test_sequences = to_categorical(y_test_sequences, num_classes=4)

        prediction_probs = []
        model_map = {"LSTM": lstm_model, "GRU": gru_model, "Transformer": transformer_model, "AST": ast_model}
        audio_res = model_map[config.audio_model]("audio", X_train_sequences_audio, X_val_sequences_audio, y_train, y_train_sequences, X_test_sequences_audio, y_val, y_test, y_val_sequences, config)
        facial_res = model_map[config.facial_model]("facial", X_train_sequences_facial, X_val_sequences_facial, y_train, y_train_sequences, X_test_sequences_facial, y_val, y_test, y_val_sequences, config)
        pose_res = model_map[config.pose_model]("pose", X_train_sequences_pose, X_val_sequences_pose, y_train, y_train_sequences, X_test_sequences_pose, y_val, y_test, y_val_sequences, config)

        prediction_probs.extend([audio_res, facial_res, pose_res])  # prediction probability vectors
        
        final_prediction = np.mean(prediction_probs, axis=0)
        y_pred = np.argmax(final_prediction, axis=1)
        
        y_test_sequences = np.argmax(y_test_sequences, axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)

        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

            wandb.log({f"fold_{fold}_metrics": test_metrics})
            print(f"Fold {fold} Test Metrics:", test_metrics)

        avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
        wandb.run.summary.update(avg_test_metrics)
        
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def main():

    sweep_config = {
        'method': 'random',
        'name': 'model_v0_TEST',
        'parameters': {
            'audio_model': {'values': ['Transformer']}, #['LSTM', 'GRU', 'Transformer', 'AST']},
            'facial_model': {'values': ['Transformer']}, #['LSTM', 'GRU', 'Transformer']},
            'pose_model': {'values': ['Transformer']}, #['LSTM', 'GRU', 'Transformer']},

            'use_pca': {'values': [True, False]},

            'gru_use_bidirectional': {'values': [True, False]},
            'gru_num_layers': {'values': [1, 2, 3]},
            'gru_units': {'values': [64, 128, 256]},

            'lstm_use_bidirectional': {'values': [True, False]},
            'lstm_num_layers': {'values': [1, 2, 3]},
            'lstm_units': {'values': [64, 128, 256]},

            'transformer_num_heads': {'values': [4, 8, 16]},
            'transformer_num_layers': {'values': [2, 4, 6]},
            'transformer_hidden_dim': {'values': [64, 128, 256]},

            'ast_num_heads': {'values': [2, 4, 8, 16, 32, 100]},
            'ast_num_layers': {'values': [2, 3]},
            'ast_hidden_dim': {'values': [64, 128, 256]},

            'dropout': {'values': [0.0, 0.3, 0.5, 0.8]},
            'dense_units': {'values': [32, 64, 128]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid', 'softmax']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'RMSprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005, 0.0005]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'value': 5},
            'loss': {'values': ["categorical_crossentropy"]},
            'reg': {'values': ['l1', 'l2', 'l1_l2']},
            'sequence_length': {'values': [30, 60, 90]},
        }
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="model_v0_TEST")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
