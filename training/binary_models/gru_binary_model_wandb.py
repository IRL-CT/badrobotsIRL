import wandb
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2, l1, l2
from keras.utils import to_categorical
import tensorflow as tf
from create_data_splits import create_data_splits, create_data_splits_pca
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
    data = config.data

    if data == "reg" or data == "norm":
        splits = create_data_splits(
                df, "binary",
                fold_no=0,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
        if splits is None:
            return

    if data == "pca":
        splits = create_data_splits_pca(
                df, "binary",
                fold_no=0,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
        if splits is None:
            return

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

    if loss == "categorical_crossentropy":
        y_train_sequences = to_categorical(y_train_sequences, num_classes=2)
        y_val_sequences = to_categorical(y_val_sequences, num_classes=2)
        y_test_sequences = to_categorical(y_test_sequences, num_classes=2)

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
    data = config.data
    feature_set = config.feature_set

    splits_pose, splits_facial, splits_audio = None, None, None

    if feature_set == "full":
        if data in ["reg", "norm"]:
            splits_pose = create_data_splits(df_pose, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        elif data == "pca":
            splits_pose = create_data_splits_pca(df_pose, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits_pca(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits_pca(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_pose is None: raise ValueError(f"Failed to create splits for pose modality.")
        if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
        if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")

    elif feature_set == "stats":
        if data in ["reg", "norm"]:
            splits_facial = create_data_splits(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        elif data == "pca":
            splits_facial = create_data_splits_pca(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits_pca(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
        if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")

    if splits_pose is not None:
        X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
    else:
        X_train_pose = X_val_pose = X_test_pose = X_train_pose_seq = X_val_pose_seq = X_test_pose_seq = None

    X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
    X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio

    if loss == "categorical_crossentropy":
        y_train_sequences = to_categorical(y_train_sequences, num_classes=2)
        y_val_sequences = to_categorical(y_val_sequences, num_classes=2)
        y_test_sequences = to_categorical(y_test_sequences, num_classes=2)

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
    data = config.data
    feature_set = config.feature_set

    splits_pose, splits_facial, splits_audio = None, None, None

    if feature_set == "full":
        if data in ["reg", "norm"]:
            splits_pose = create_data_splits(df_pose, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        elif data == "pca":
            splits_pose = create_data_splits_pca(df_pose, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_facial = create_data_splits_pca(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits_pca(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_pose is None: raise ValueError(f"Failed to create splits for pose modality.")
        if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
        if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")

    elif feature_set == "stats":
        if data in ["reg", "norm"]:
            splits_facial = create_data_splits(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
        elif data == "pca":
            splits_facial = create_data_splits_pca(df_facial, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
            splits_audio = create_data_splits_pca(df_audio, "binary", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

        if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
        if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")

    if splits_pose is not None:
        X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
    else:
        X_train_pose = X_val_pose = X_test_pose = X_train_pose_seq = X_val_pose_seq = X_test_pose_seq = None

    X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
    X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio

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
    
    feature_set = config.feature_set
    modality = config.modality
    data = config.data
    fusion_type = config.fusion_type

    df = pd.read_csv("../../preprocessing/merged_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_merged_correct_stats_0_3.csv")

    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = df.iloc[:, 28:63]
    df_audio_index = df.iloc[:, 63:]

    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]

    print("--------\nfeature set: ", feature_set, "\nmodality: ", modality, "\ndata: ", data, "\nfusion: ", fusion_type, "\n--------")

    modality_mapping = {
        "pose": pd.concat([info, df_pose_index], axis=1),
        "facial": pd.concat([info, df_facial_index], axis=1),
        "audio": pd.concat([info, df_audio_index], axis=1),
        "pose_facial": pd.concat([info, df_pose_index, df_facial_index], axis=1),
        "pose_audio": pd.concat([info, df_pose_index, df_audio_index], axis=1),
        "facial_audio": pd.concat([info, df_facial_index, df_audio_index], axis=1),
        "combined": pd.concat([info, df_pose_index, df_facial_index, df_audio_index], axis=1),
    }

    modality_mapping_stats = {
        "facial": pd.concat([info, df_facial_index_stats], axis=1),
        "audio": pd.concat([info, df_audio_index_stats], axis=1),
        "facial_audio": pd.concat([info, df_facial_index_stats, df_audio_index_stats], axis=1),
        "combined": pd.concat([info, df_facial_index_stats, df_audio_index_stats], axis=1),
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
            if "pose" in modality:
                print("pose not in stats")
                wandb.finish()
            else:
                df = modality_mapping_stats.get(modality)

        if data != "reg":
            df = create_normalized_df(df)

        return df

    if feature_set == "full":
        df = get_modality_data(modality, data)

        if fusion_type == "early" and modality == "combined":
            train_early_fusion(df, config)
        elif fusion_type == "intermediate" and modality == "combined":
            train_intermediate_fusion(
                get_modality_data("pose", data),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        elif fusion_type == "late" and modality == "combined":
            train_late_fusion(
                get_modality_data("pose", data),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        else:
            train_single_modality_model(get_modality_data(modality, data), config)
        
    elif feature_set == "stats":
        df = get_modality_data(modality, data)

        if fusion_type == "early" and modality == "combined":
            train_early_fusion(df, config)
        elif fusion_type == "intermediate" and modality == "combined":
            train_intermediate_fusion(
                pd.DataFrame(),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        elif fusion_type == "late" and modality == "combined":
            train_late_fusion(
                pd.DataFrame(),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        else:
            train_single_modality_model(get_modality_data(modality, data), config)

def main():
    
    sweep_config = {
        'method': 'random',
        'name': 'gru_all_v1',
        'parameters': {
            'feature_set' : {'values': ["full", "stats"]},
            'modality' : {'values': ['combined', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio']},
            'data' : {'values' : ["reg", "norm", "pca"]},
            'fusion_type': {'values': ['early', 'intermediate', 'late']},
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
            'sequence_length' : {'values' : [30, 60, 90]}
        }
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="gru_all_v1")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
