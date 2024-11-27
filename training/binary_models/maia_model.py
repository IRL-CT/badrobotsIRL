import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

from create_data_splits import create_data_splits

import wandb

def model(df):

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

    activation = config.activation
    dropout = config.dropout
    optimizer = config.optimizer
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    epochs = config.epochs
    loss = 'categorical_crossentropy'

    splits = create_data_splits(df, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    input_shape = (X_train_sequences.shape[2],)

    model = Sequential()

    model.add(Dense(units=64, input_shape=input_shape, activation=activation))
    model.add(Dense(units=128, activation=activation))
    model.add(Dense(units=64, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    model_history = model.fit(
        np.array(X_train),
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 2,
        validation_data=(np.array(X_val), y_val)
    )

    for epoch in range(epochs):
        metrics = {
            'epoch': epoch,
            'loss': model_history.history['loss'][epoch],
            'val_loss': model_history.history['val_loss'][epoch],
            'accuracy': model_history.history.get('accuracy')[epoch],
            'val_accuracy': model_history.history.get('val_accuracy')[epoch],
            'precision': model_history.history.get('precision')[epoch],
            'val_precision': model_history.history.get('val_precision')[epoch],
            'recall': model_history.history.get('recall')[epoch],
            'val_recall': model_history.history.get('val_recall')[epoch],
            'auc': model_history.history.get('auc')[epoch],
            'val_auc': model_history.history.get('val_auc')[epoch]
        }
        wandb.log(metrics)

    test_loss, test_accuracy = model.evaluate(np.array(X_test), y_test)

    y_predict_probs = model.predict(np.array(X_test))
    y_predict = np.argmax(y_predict_probs, axis=1)

    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    })

    wandb.finish()


def main():
    df = pd.read_csv("preprocessing/merged_features/all_participants_merged_correct_normalized.csv")

    sweep_config = {
        'method': 'random',
        'name' : 'maia_model',
        'parameters' : {
            'activation': {'values' : ['sigmoid', 'relu', 'softmax']},
            'dropout' : {'values' : [0, 0.2, 0.6]},
            'optimizer' : {'values' : ['sgd', 'adam']},
            'batch_size' : {'values' : [256, 512, 1024]},
            'sequence_length' : {'values' : [15, 30, 90]},
            'epochs': {'value' : 500}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="maia_model")
    wandb.agent(sweep_id, function=lambda: model(df))
    

if __name__ == '__main__':
    main()
