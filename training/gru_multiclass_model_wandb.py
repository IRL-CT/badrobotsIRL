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
from create_data_splits import create_multiclass_data_splits, create_data_splits_pca
from get_metrics import get_test_metrics
from gru_single_modality import train_single_modality_model

def build_early_late_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg):
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

    return model

def train_early_fusion(df, config):

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
    use_pca = config.use_pca

    splits = create_multiclass_data_splits(
            df,
            fold_no=0,
            num_folds=5,
            seed_value=42,
            sequence_length=sequence_length)

    # if use_pca == True:
    #     splits = create_data_splits_pca(
    #     df,
    #     fold_no=0,
    #     num_folds=5,
    #     seed_value=42,
    #     sequence_length=sequence_length
    #     )
    # else:
    #     splits = create_data_splits(
    #         df,
    #         fold_no=0,
    #         num_folds=5,
    #         seed_value=42,
    #         sequence_length=sequence_length
    #     )

    if splits is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

    if loss == "categorical_crossentropy":
        y_train_sequences = to_categorical(y_train_sequences)
        y_val_sequences = to_categorical(y_val_sequences)
        y_test_sequences = to_categorical(y_test_sequences)

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

    model = build_early_late_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg)
    
    if loss == "categorical_crossentropy":
        num_classes = len(np.unique(y_train_sequences))
        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(num_classes, activation="softmax"))
    else:
        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(1, activation="sigmoid"))

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.summary()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

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
    
    if loss == "categorical_crossentropy":
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)
    else:
        y_pred = (y_predict_probs > 0.5).astype(int).flatten()
        y_test_sequences = y_test_sequences.astype(int).flatten()

    test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


def train_intermediate_fusion(df_pose, df_facial, df_audio, config):

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

    splits_pose = create_multiclass_data_splits(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
    splits_facial = create_multiclass_data_splits(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
    splits_audio = create_multiclass_data_splits(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

    if splits_pose is None or splits_facial is None or splits_audio is None:
        return
    
    X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
    X_train_facial, _, _, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
    X_train_audio, _, _, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio


    if loss == "categorical_crossentropy":
        y_train_sequences = to_categorical(y_train_sequences)
        y_val_sequences = to_categorical(y_val_sequences)
        y_test_sequences = to_categorical(y_test_sequences)

    print("X_train_pose_seq shape:", X_train_pose_seq.shape)
    print("X_val_pose_seq shape:", X_val_pose_seq.shape)
    print("X_test_pose_seq shape:", X_test_pose_seq.shape)

    feature_inputs = [
        Input(shape=(sequence_length, X_train_pose_seq.shape[2])),
        Input(shape=(sequence_length, X_train_facial_seq.shape[2])),
        Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
    ]
    feature_outputs = []

    for feature_input in feature_inputs:
        x = feature_input
        for _ in range(num_gru_layers):
            x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=kernel_regularizer)(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        feature_outputs.append(x)

    concatenated_features = concatenate(feature_outputs)

    x = GRU(gru_units, activation=activation, kernel_regularizer=kernel_regularizer)(concatenated_features)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    if loss == "categorical_crossentropy":
        num_classes = len(np.unique(y_train_sequences))
        x = Dense(dense_units, activation=activation)(x)
        x = Dense(num_classes, activation="softmax")(x)
    else:
        x = Dense(dense_units, activation=activation)(x)
        x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=feature_inputs, outputs=x)

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.summary()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    model_checkpoint = ModelCheckpoint("../best_model.keras", monitor="val_accuracy", save_best_only=True)

    model_history = model.fit(
        [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
        # callbacks=[model_checkpoint],
        verbose=2
    )

    for epoch in range(len(model_history.history['loss'])):

        metrics = {'epoch': epoch + 1}

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

    y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
    
    if loss == "categorical_crossentropy":
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)
    else:
        y_pred = (y_predict_probs > 0.5).astype(int).flatten()
        y_test_sequences = y_test_sequences.astype(int).flatten()

    test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


def train_late_fusion(df_pose, df_facial, df_audio, config):

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
    use_pca = config.use_pca

    splits_pose = create_multiclass_data_splits(df_pose, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
    splits_facial = create_multiclass_data_splits(df_facial, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
    splits_audio = create_multiclass_data_splits(df_audio, fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

    if splits_pose is None or splits_facial is None or splits_audio is None:
        return
    
    X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
    X_train_facial, _, _, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
    X_train_audio, _, _, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

    pose_input = Input(shape=(sequence_length, X_train_pose_seq.shape[2]))
    facial_input = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
    audio_input = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))

    pose_model = build_early_late_model(sequence_length, X_train_pose_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
    facial_model = build_early_late_model(sequence_length, X_train_facial_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
    audio_model = build_early_late_model(sequence_length, X_train_audio_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)

    pose_output = pose_model(pose_input)
    facial_output = facial_model(facial_input)
    audio_output = audio_model(audio_input)

    concatenated = concatenate([pose_output, facial_output, audio_output])

    if loss == "categorical_crossentropy":
        num_classes = len(np.unique(y_train_sequences))
        x = Dense(dense_units, activation=activation)(concatenated)
        output = Dense(num_classes, activation="softmax")(x)
    else:
        x = Dense(dense_units, activation=activation)(concatenated)
        output = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[pose_input, facial_input, audio_input], outputs=output)

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    
    model_checkpoint = ModelCheckpoint("../best_model.keras", monitor="val_accuracy", save_best_only=True)

    model_history = model.fit(
        [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
        # callbacks=[model_checkpoint],
        verbose=2
    )

    for epoch in range(len(model_history.history['loss'])):

        metrics = {'epoch': epoch + 1}

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

    y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
    
    if config.loss == "categorical_crossentropy":
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)
    else:
        y_pred = (y_predict_probs > 0.5).astype(int).flatten()
        y_test_sequences = y_test_sequences.astype(int).flatten()

    test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


def train():

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    use_pca = config.use_pca
    use_norm = config.use_norm
    fusion_type = config.fusion_type
    modality = config.modality

    df = pd.read_csv("../preprocessing/merged_features/all_participants_merged_correct.csv")
    df_norm = pd.read_csv("../preprocessing/merged_features/all_participants_merged_correct_normalized.csv")
    df_norm_pca = pd.read_csv("../preprocessing/merged_features/all_participants_merged_correct_normalized_principal.csv")

    df_pose = pd.read_csv("../preprocessing/individual_features/all_participants_pose_features.csv")
    df_pose_norm = pd.read_csv("../preprocessing/individual_features/all_participants_pose_features_norm.csv")
    df_pose_norm_pca = pd.read_csv("../preprocessing/individual_features/all_participants_pose_features_norm_pca.csv")

    df_facial = pd.read_csv("../preprocessing/individual_features/all_participants_facial_features.csv")
    df_facial_norm = pd.read_csv("../preprocessing/individual_features/all_participants_facial_features_norm.csv")
    df_facial_norm_pca = pd.read_csv("../preprocessing/individual_features/all_participants_facial_features_norm_pca.csv")

    df_audio = pd.read_csv("../preprocessing/individual_features/all_participants_audio_features.csv")
    df_audio_norm = pd.read_csv("../preprocessing/individual_features/all_participants_audio_features_norm.csv")
    df_audio_norm_pca = pd.read_csv("../preprocessing/individual_features/all_participants_audio_features_norm_pca.csv")

    df_pose_facial = pd.read_csv("../preprocessing/semi_merged_features/all_participants_pose_facial_features.csv")
    df_pose_facial_norm = pd.read_csv("../preprocessing/semi_merged_features/all_participants_pose_facial_features_norm.csv")
    df_pose_facial_norm_pca = pd.read_csv("../preprocessing/semi_merged_features/all_participants_pose_facial_features_norm_pca.csv")

    df_pose_audio = pd.read_csv("../preprocessing/semi_merged_features/all_participants_pose_audio_features.csv")
    df_pose_audio_norm = pd.read_csv("../preprocessing/semi_merged_features/all_participants_pose_audio_features_norm.csv")
    df_pose_audio_norm_pca = pd.read_csv("../preprocessing/semi_merged_features/all_participants_pose_audio_features_norm_pca.csv")

    df_facial_audio = pd.read_csv("../preprocessing/semi_merged_features/all_participants_facial_audio_features.csv")
    df_facial_audio_norm = pd.read_csv("../preprocessing/semi_merged_features/all_participants_facial_audio_features_norm.csv")
    df_facial_audio_norm_pca = pd.read_csv("../preprocessing/semi_merged_features/all_participants_facial_audio_features_norm_pca.csv")

    if modality == "combined":
        if use_norm:
            if use_pca:
                if fusion_type == 'early':
                    train_early_fusion(df_norm_pca, config)

                elif fusion_type == 'intermediate':
                    train_intermediate_fusion(df_pose_norm_pca, df_facial_norm_pca, df_audio_norm_pca, config)

                elif fusion_type == 'late':
                    train_late_fusion(df_pose_norm_pca, df_facial_norm_pca, df_audio_norm_pca, config)

            else:
                if fusion_type == 'early':
                    train_early_fusion(df_norm, config)

                elif fusion_type == 'intermediate':
                    train_intermediate_fusion(df_pose_norm, df_facial_norm, df_audio_norm, config)

                elif fusion_type == 'late':
                    train_late_fusion(df_pose_norm, df_facial_norm, df_audio_norm, config)

        else:
            if fusion_type == 'early':
                train_early_fusion(df, config)

            elif fusion_type == 'intermediate':
                train_intermediate_fusion(df_pose, df_facial, df_audio, config)

            elif fusion_type == 'late':
                train_late_fusion(df_pose, df_facial, df_audio, config)

    elif modality == "pose":
        if use_norm:
            if use_pca:
                train_single_modality_model(df_pose_norm_pca, config)
            else:
                train_single_modality_model(df_pose_norm, config)
        else:
            train_single_modality_model(df_pose, config)

    elif modality == "facial":
        if use_norm:
            if use_pca:
                train_single_modality_model(df_facial_norm_pca, config)
            else:
                train_single_modality_model(df_facial_norm, config)
        else:
            train_single_modality_model(df_facial, config)

    elif modality == "audio":
        if use_norm:
            if use_pca:
                train_single_modality_model(df_audio_norm_pca, config)
            else:
                train_single_modality_model(df_audio_norm, config)
        else:
            train_single_modality_model(df_audio, config)

    elif modality == "pose_facial":
        if use_norm:
            if use_pca:
                train_single_modality_model(df_pose_facial_norm_pca, config)
            else:
                train_single_modality_model(df_pose_facial_norm, config)
        else:
            train_single_modality_model(df_pose_facial, config)

    elif modality == "pose_audio":
        if use_norm:
            if use_pca:
                train_single_modality_model(df_pose_audio_norm_pca, config)
            else:
                train_single_modality_model(df_pose_audio_norm, config)
        else:
            train_single_modality_model(df_pose_audio, config)

    elif modality == "facial_audio":
        if use_norm:
            if use_pca:
                train_single_modality_model(df_facial_audio_norm_pca, config)
            else:
                train_single_modality_model(df_facial_audio_norm, config)
        else:
            train_single_modality_model(df_facial_audio, config)

def main():
    
    sweep_config = {
        'method': 'random',
        'name': 'gru_multiclass_all_v0',
        'parameters': {
            'use_pca': {'values': [True, False]},
            'use_norm' : {'values' : [True, False]},
            'use_bidirectional': {'values': [True, False]},
            'num_gru_layers': {'values': [1, 2, 3]},
            'gru_units': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'dense_units': {'values': [32, 64, 128]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'value': 500},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss' : {'values' : ["categorical_crossentropy"]},
            'sequence_length' : {'values' : [30, 60, 90]},
            'fusion_type': {'values': ['early', 'intermediate', 'late']},
            'modality' : {'values': ['combined', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio']}
        }
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="gru_multiclass_all_v0")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
