import wandb
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2, l1, l2
from keras.utils import to_categorical
import tensorflow as tf
from create_data_splits import create_multiclass_data_splits
from get_metrics import get_test_metrics

def train_single_modality_model(df, config):

    print("training model")

    num_gru_layers = config.num_gru_layers
    gru_units = config.gru_units
    batch_size = config.batch_size
    epochs = config.epochs
    activation = config.activation_function
    use_bidirectional = config.use_bidirectional
    dropout = config.dropout_rate
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
    kernel_regularizer = config.recurrent_regularizer
    loss = config.loss
    sequence_length = config.sequence_length

    splits = create_multiclass_data_splits(
    df,
    fold_no=0,
    num_folds=5,
    seed_value=42,
    sequence_length=sequence_length
    )

    if splits is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

    y_train_sequences = to_categorical(y_train_sequences, num_classes=6)
    y_val_sequences = to_categorical(y_val_sequences, num_classes=6)
    y_test_sequences = to_categorical(y_test_sequences, num_classes=6)

    print("X_train_sequences shape:", X_train_sequences.shape)
    print("X_val_sequences shape:", X_val_sequences.shape)
    print("X_test_sequences shape:", X_test_sequences.shape)

    if kernel_regularizer == "l1":
        reg = l1(0.01)
    elif kernel_regularizer == "l2":
        reg = l2(0.01)
    elif kernel_regularizer == "l1_l2":
        reg = l1_l2(0.01, 0.01)
    else:
        reg = None
    
    input_shape = X_train_sequences.shape[2]

    model = Sequential()
    model.add(Input(shape=(sequence_length, input_shape)))

    if num_gru_layers == 1:
        if use_bidirectional:
            model.add(Bidirectional(GRU(gru_units, activation=activation, kernel_regularizer=reg)))
        else:
            model.add(GRU(gru_units, activation=activation, kernel_regularizer=reg))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
    else:
        for _ in range(num_gru_layers - 1):
            if use_bidirectional:
                model.add(Bidirectional(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)))
            else:
                model.add(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())

        if use_bidirectional:
            model.add(Bidirectional(GRU(gru_units, activation=activation)))
        else:
            model.add(GRU(gru_units, activation=activation))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
    
    num_classes = len(np.unique(y_train_sequences))
    print("Num classes: ", num_classes)
    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_val:", np.unique(y_val))
    print("Unique labels in y_test:", np.unique(y_test))

    model.add(Dense(dense_units, activation=activation))
    model.add(Dense(num_classes, activation="softmax"))
    print(f"Model output classes: {num_classes}")
    
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.summary()
    
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    model_checkpoint = ModelCheckpoint("../best_model.keras", monitor="val_accuracy", save_best_only=True)
    
    model_history = model.fit(
        X_train_sequences, y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_sequences, y_val_sequences),
        # callbacks=[model_checkpoint],
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
    
    y_pred = np.argmax(y_predict_probs, axis=1)
    y_test_sequences = np.argmax(y_test_sequences, axis=1)

    test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)

# def train():

#     wandb.init()
#     config = wandb.config
#     print(config)

#     seed_value = 42
#     np.random.seed(seed_value)
#     random.seed(seed_value)
#     tf.random.set_seed(seed_value)

#     use_pca = config.use_pca
#     use_norm = config.use_norm

#     df = pd.read_csv("../preprocessing/individual_features/all_participants_pose_features.csv")
#     df_norm = pd.read_csv("../preprocessing/individual_features/all_participants_pose_features_norm.csv")
#     df_norm_pca = pd.read_csv("../preprocessing/individual_features/all_participants_pose_features_norm_pca.csv")

#     if use_norm:
#         if use_pca:
#             train_single_modality_model(df_norm_pca, config)
#         else:
#             train_single_modality_model(df_norm, config)
#     else:
#         train_single_modality_model(df, config)

# def main():
#     sweep_config = {
#         'method': 'random',
#         'name': 'pose_features_gru_v2',
#         'parameters': {
#             'use_pca': {'values': [True, False]},
#             'use_norm': {'values': [True, False]},
#             'use_bidirectional': {'values': [True, False]},
#             'num_gru_layers': {'values': [1, 2, 3]},
#             'gru_units': {'values': [64, 128, 256]},
#             'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
#             'dense_units': {'values': [32, 64, 128]},
#             'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
#             'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
#             'learning_rate': {'values': [0.001, 0.01, 0.005]},
#             'batch_size': {'values': [32, 64, 128]},
#             'epochs': {'value': 500},
#             'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
#             'loss' : {'values' : ["binary_crossentropy", "categorical_crossentropy"]},
#             'sequence_length' : {'values' : [30, 60, 90]},
#         }
#     }

#     print(sweep_config)

#     def train_wrapper():
#         train()

#     sweep_id = wandb.sweep(sweep=sweep_config, project="pose_features_gru_v2")
#     wandb.agent(sweep_id, function=train_wrapper)

# if __name__ == '__main__':
#     main()
