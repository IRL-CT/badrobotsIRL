import wandb

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1_l2
import tensorflow as tf

from create_data_splits import create_data_splits
from get_metrics import get_metrics


def train():
    wandb.init()
    config = wandb.config

    df = pd.read_csv("preprocessing/merged_features/all_participants_normalized.csv")
    df['fold_id'] = -1
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        df.loc[test_index, 'fold_id'] = fold + 1

    with_val = 1
    fold_no = 1

    splits = create_data_splits(
        df,
        fold_no=fold_no,
        with_val=with_val,
        fold_col='fold_id',
        seed_value=42,
        sequence_length=1
    )

    if splits is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences = splits
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    X_test_sequences = np.array(X_test_sequences, dtype=float)

    model = Sequential()
    model.add(Input(shape=(X_train_sequences.shape[1], X_train_sequences.shape[2])))

    # Add (Bidirectional) LSTM layers
    for _ in range(config.num_lstm_layers):
        if config.use_bidirectional:
            model.add(Bidirectional(LSTM(config.lstm_units, return_sequences=True, activation='tanh', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))))
        else:
            model.add(LSTM(config.lstm_units, return_sequences=True, activation='tanh', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(config.dropout_rate))
        model.add(BatchNormalization())

    if config.use_bidirectional:
        model.add(Bidirectional(LSTM(config.lstm_units, activation='tanh')))
    else:
        model.add(LSTM(config.lstm_units, activation='tanh'))
    model.add(Dropout(config.dropout_rate))
    model.add(BatchNormalization())
    
    # Add Dense layers
    model.add(Dense(config.dense_units, activation=config.activation_function))
    model.add(Dense(config.dense_units, activation=config.activation_function))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = config.optimizer.lower()
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("training/best_model.keras", monitor="val_accuracy", save_best_only=True)
    
    model_history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, early_stopping],
        verbose=2
    )

    for epoch in range(len(model_history.history['loss'])):
        metrics = {
            'epoch': epoch,
            'loss': model_history.history['loss'][epoch],
            'val_loss': model_history.history['val_loss'][epoch]
        }
        if 'accuracy' in model_history.history:
            metrics['accuracy'] = model_history.history['accuracy'][epoch]
        if 'val_accuracy' in model_history.history:
            metrics['val_accuracy'] = model_history.history['val_accuracy'][epoch]
        if 'precision' in model_history.history:
            metrics['precision'] = model_history.history['precision'][epoch]
        if 'val_precision' in model_history.history:
            metrics['val_precision'] = model_history.history['val_precision'][epoch]
        if 'recall' in model_history.history:
            metrics['recall'] = model_history.history['recall'][epoch]
        if 'val_recall' in model_history.history:
            metrics['val_recall'] = model_history.history['val_recall'][epoch]
        if 'auc' in model_history.history:
            metrics['auc'] = model_history.history['auc'][epoch]
        if 'val_auc' in model_history.history:
            metrics['val_auc'] = model_history.history['val_auc'][epoch]
        
        wandb.log(metrics)

    y_predict_probs = model.predict(X_test_sequences)
    y_pred = (y_predict_probs > 0.5).astype(int).flatten()
    y_test_sequences = y_test_sequences.astype(int).flatten()

    test_metrics = get_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


sweep_config = {
    'method': 'grid',
    'name': 'lstm_vs_bilstm_sweep',
    'parameters': {
        'use_bidirectional': {'values': [False]},
        'num_lstm_layers': {'values': [1, 2, 3]},
        'lstm_units': {'values': [64, 128, 256]},
        'dropout_rate': {'values': [0.2, 0.3, 0.4]},
        'dense_units': {'values': [32, 64, 128]},
        'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
        'optimizer': {'values': ['adam', 'sgd']},
        'learning_rate': {'values': [0.001, 0.01, 0.005]},
        'batch_size': {'values': [64, 128]},
        'epochs': {'value': 100},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="lstm_vs_bilstm_comparison")
wandb.agent(sweep_id, function=train, count=100)