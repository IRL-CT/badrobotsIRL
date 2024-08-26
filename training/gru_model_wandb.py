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
from create_data_splits import create_data_splits, create_data_splits_pca
from get_metrics import get_metrics

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

    if use_pca == True:
        splits = create_data_splits_pca(
        df,
        fold_no=0,
        num_folds=5,
        seed_value=42,
        sequence_length=sequence_length
        )
    else:
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

    model_checkpoint = ModelCheckpoint("training/best_model.keras", monitor="val_accuracy", save_best_only=True)
    
    model_history = model.fit(
        X_train_sequences, y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_sequences, y_val_sequences),
        callbacks=[model_checkpoint],
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

    test_metrics = get_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


def train_intermediate_fusion(df, config):

    df_pose = df.iloc[:, 4:29]
    df_facial = df.iloc[:, 29:65]
    df_audio = df.iloc[:, 65:]

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

    model_checkpoint = ModelCheckpoint("training/best_model.keras", monitor="val_accuracy", save_best_only=True)

    model_history = model.fit(
        [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
        callbacks=[model_checkpoint],
        verbose=2
    )

    for epoch in range(len(model_history.history['loss'])):

        metrics = {'epoch': epoch + 1}

        if 'loss' in model_history.history:
            metrics['total_train_loss'] = model_history.history['loss'][epoch]
        if 'val_loss' in model_history.history:
            metrics['total_val_loss'] = model_history.history['val_loss'][epoch]

        for i, feature_name in enumerate(["pose", "facial", "audio"]):
            if f'loss_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_train_loss'] = model_history.history[f'loss_{feature_name}'][epoch]
            if f'val_loss_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_val_loss'] = model_history.history[f'val_loss_{feature_name}'][epoch]

            if 'accuracy' in model_history.history:
                metrics[f'{feature_name}_train_accuracy'] = model_history.history[f'accuracy_{feature_name}'][epoch]
            if 'val_accuracy' in model_history.history:
                metrics[f'{feature_name}_val_accuracy'] = model_history.history[f'val_accuracy_{feature_name}'][epoch]
            
            if 'precision' in model_history.history:
                metrics[f'{feature_name}_train_precision'] = model_history.history[f'precision_{feature_name}'][epoch]
            if 'val_precision' in model_history.history:
                metrics[f'{feature_name}_val_precision'] = model_history.history[f'val_precision_{feature_name}'][epoch]
            
            if 'recall' in model_history.history:
                metrics[f'{feature_name}_train_recall'] = model_history.history[f'recall_{feature_name}'][epoch]
            if 'val_recall' in model_history.history:
                metrics[f'{feature_name}_val_recall'] = model_history.history[f'val_recall_{feature_name}'][epoch]
            
            if 'auc' in model_history.history:
                metrics[f'{feature_name}_train_auc'] = model_history.history[f'auc_{feature_name}'][epoch]
            if 'val_auc' in model_history.history:
                metrics[f'{feature_name}_val_auc'] = model_history.history[f'val_auc_{feature_name}'][epoch]

        wandb.log(metrics)

    y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
    
    if loss == "categorical_crossentropy":
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)
    else:
        y_pred = (y_predict_probs > 0.5).astype(int).flatten()
        y_test_sequences = y_test_sequences.astype(int).flatten()

    test_metrics = get_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


def train_late_fusion(df, config):

    df_pose = df.iloc[:, 4:29]
    df_facial = df.iloc[:, 29:65]
    df_audio = df.iloc[:, 65:]

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

    pose_model = build_early_late_model(sequence_length, X_train_pose_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
    facial_model = build_early_late_model(sequence_length, X_train_facial_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
    audio_model = build_early_late_model(sequence_length, X_train_audio_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)

    pose_output = pose_model(X_train_pose_seq)
    facial_output = facial_model(X_train_facial_seq)
    audio_output = audio_model(X_train_audio_seq)

    concatenated = concatenate([pose_output, facial_output, audio_output])

    if loss == "categorical_crossentropy":
        num_classes = len(np.unique(y_train_sequences))
        x = Dense(dense_units, activation=activation)(concatenated)
        output = Dense(num_classes, activation="softmax")(x)
    else:
        x = Dense(dense_units, activation=activation)(concatenated)
        output = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[pose_model.input, facial_model.input, audio_model.input], outputs=output)

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    
    model_checkpoint = ModelCheckpoint("training/best_model.keras", monitor="val_accuracy", save_best_only=True)

    model_history = model.fit(
        [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
        callbacks=[model_checkpoint],
        verbose=2
    )

    for epoch in range(len(model_history.history['loss'])):

        metrics = {'epoch': epoch + 1}

        if 'loss' in model_history.history:
            metrics['total_train_loss'] = model_history.history['loss'][epoch]
        if 'val_loss' in model_history.history:
            metrics['total_val_loss'] = model_history.history['val_loss'][epoch]
        if 'accuracy' in model_history.history:
            metrics['total_train_accuracy'] = model_history.history['accuracy'][epoch]
        if 'val_accuracy' in model_history.history:
            metrics['total_val_accuracy'] = model_history.history['val_accuracy'][epoch]
        if 'precision' in model_history.history:
            metrics['total_train_precision'] = model_history.history['precision'][epoch]
        if 'val_precision' in model_history.history:
            metrics['total_val_precision'] = model_history.history['val_precision'][epoch]
        if 'recall' in model_history.history:
            metrics['total_train_recall'] = model_history.history['recall'][epoch]
        if 'val_recall' in model_history.history:
            metrics['total_val_recall'] = model_history.history['val_recall'][epoch]
        if 'auc' in model_history.history:
            metrics['total_train_auc'] = model_history.history['auc'][epoch]
        if 'val_auc' in model_history.history:
            metrics['total_val_auc'] = model_history.history['val_auc'][epoch]

        for i, feature_name in enumerate(["pose", "facial", "audio"]):
            if f'loss_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_train_loss'] = model_history.history[f'loss_{feature_name}'][epoch]
            if f'val_loss_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_val_loss'] = model_history.history[f'val_loss_{feature_name}'][epoch]
            if f'accuracy_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_train_accuracy'] = model_history.history[f'accuracy_{feature_name}'][epoch]
            if f'val_accuracy_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_val_accuracy'] = model_history.history[f'val_accuracy_{feature_name}'][epoch]
            if f'precision_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_train_precision'] = model_history.history[f'precision_{feature_name}'][epoch]
            if f'val_precision_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_val_precision'] = model_history.history[f'val_precision_{feature_name}'][epoch]
            if f'recall_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_train_recall'] = model_history.history[f'recall_{feature_name}'][epoch]
            if f'val_recall_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_val_recall'] = model_history.history[f'val_recall_{feature_name}'][epoch]
            if f'auc_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_train_auc'] = model_history.history[f'auc_{feature_name}'][epoch]
            if f'val_auc_{feature_name}' in model_history.history:
                metrics[f'{feature_name}_val_auc'] = model_history.history[f'val_auc_{feature_name}'][epoch]

        wandb.log(metrics)

    y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
    
    if config.loss == "categorical_crossentropy":
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)
    else:
        y_pred = (y_predict_probs > 0.5).astype(int).flatten()
        y_test_sequences = y_test_sequences.astype(int).flatten()

    test_metrics = get_metrics(y_pred, y_test_sequences, tolerance=1)
    wandb.log(test_metrics)
    print(test_metrics)


def train(df):

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

    fusion_type = config.fusion_type

    '''
    frame,participant,binary_label,multiclass_label df.iloc[:, :4]
    nose_x_delta,nose_y_delta,neck_x_delta,neck_y_delta,
    rightshoulder_x_delta,rightshoulder_y_delta,rightelbow_x_delta,rightelbow_y_delta,
    rightwrist_x_delta,rightwrist_y_delta,leftshoulder_x_delta,leftshoulder_y_delta,
    leftelbow_x_delta,leftelbow_y_delta,leftwrist_x_delta,leftwrist_y_delta,
    righteye_x_delta,righteye_y_delta,lefteye_x_delta,lefteye_y_delta,
    rightear_x_delta,rightear_y_delta,leftear_x_delta,leftear_y_delta ..24 elements
    df.iloc[:, 4:29]
    AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, 
    AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, 
    AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, 
    AU26_r, AU45_r, AU01_c, AU02_c, AU04_c, 
    AU05_c, AU06_c, AU07_c, AU09_c, AU10_c, 
    AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, 
    AU23_c, AU25_c, AU26_c, AU28_c, AU45_c ..35 elements
    df.iloc[:, 29:65]
    Loudness_sma3,alphaRatio_sma3,hammarbergIndex_sma3,
    slope0-500_sma3,slope500-1500_sma3,spectralFlux_sma3,
    mfcc1_sma3,mfcc2_sma3,mfcc3_sma3,
    mfcc4_sma3,F0semitoneFrom27.5Hz_sma3nz,jitterLocal_sma3nz,
    shimmerLocaldB_sma3nz,HNRdBACF_sma3nz,logRelF0-H1-H2_sma3nz,
    logRelF0-H1-A3_sma3nz,F1frequency_sma3nz,F1bandwidth_sma3nz,
    F1amplitudeLogRelF0_sma3nz,F2frequency_sma3nz,F2bandwidth_sma3nz,
    F2amplitudeLogRelF0_sma3nz,F3frequency_sma3nz,F3bandwidth_sma3nz,
    F3amplitudeLogRelF0_sma3nz
    df.iloc[:, 65:]
    '''

    if fusion_type == 'early':
        train_early_fusion(df, config)

    elif fusion_type == 'intermediate':
        train_intermediate_fusion(df, config)

    elif fusion_type == 'late':
        train_late_fusion(df, config)
        

def main():
    global df
    df = pd.read_csv("preprocessing/merged_features/all_participants_merged_correct_normalized.csv")

    sweep_config = {
        'method': 'random',
        'name': 'gru_sweep_v1',
        'parameters': {
            'use_pca': {'values': [True, False]},
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
            'loss' : {'values' : ["binary_crossentropy", "categorical_crossentropy"]},
            'sequence_length' : {'values' : [1, 5, 15, 30, 60, 90]},
            'fusion_type': {'values': ['early', 'intermediate', 'late']}
        }
    }

    print(sweep_config)

    def train_wrapper():
        train(df)

    sweep_id = wandb.sweep(sweep=sweep_config, project="gru_sweep_v1")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
