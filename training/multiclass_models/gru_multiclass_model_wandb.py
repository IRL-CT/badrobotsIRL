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
    part_fusion = config.part_fusion

    test_metrics_list = { "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }

    for fold in range(5):

        participant_dfs = [group for _, group in df.groupby('participant')]

        feature_inputs = []
        feature_outputs = []

        if kernel_regularizer == "l1":
                reg = l1(0.01)
        elif kernel_regularizer == "l2":
            reg = l2(0.01)
        elif kernel_regularizer == "l1_l2":
            reg = l1_l2(0.01, 0.01)
        else:
            reg = None

        if part_fusion == "early":

            if data == "reg" or data == "norm":
                splits = create_data_splits(
                        df, "multiclass",
                        fold_no=fold,
                        num_folds=5,
                        seed_value=42,
                        sequence_length=sequence_length)
                if splits is None:
                    continue

            if data == "pca":
                splits = create_data_splits_pca(
                        df, "multiclass",
                        fold_no=fold,
                        num_folds=5,
                        seed_value=42,
                        sequence_length=sequence_length)
                if splits is None:
                    continue

            X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

            y_train_sequences = to_categorical(y_train_sequences, num_classes=4)
            y_val_sequences = to_categorical(y_val_sequences, num_classes=4)
            y_test_sequences = to_categorical(y_test_sequences, num_classes=4)

            print("X_train_sequences shape:", X_train_sequences.shape)
            print("X_val_sequences shape:", X_val_sequences.shape)
            print("X_test_sequences shape:", X_test_sequences.shape)
            print("y_train_sequences shape:", y_train_sequences.shape)
            print("y_val_sequences shape:", y_val_sequences.shape)
            print("y_test_sequences shape:", y_test_sequences.shape)
        
        elif part_fusion == "intermediate" or part_fusion == "late":

            for participant_df in participant_dfs:

                if data == "reg" or data == "norm":
                    splits = create_data_splits(
                            participant_df, "multiclass",
                            fold_no=fold,
                            num_folds=5,
                            seed_value=42,
                            sequence_length=sequence_length)
                    if splits is None:
                        continue

                if data == "pca":
                    splits = create_data_splits_pca(
                            participant_df, "multiclass",
                            fold_no=fold,
                            num_folds=5,
                            seed_value=42,
                            sequence_length=sequence_length)
                    if splits is None:
                        continue

                X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

                y_train_sequences = to_categorical(y_train_sequences, num_classes=4)
                y_val_sequences = to_categorical(y_val_sequences, num_classes=4)
                y_test_sequences = to_categorical(y_test_sequences, num_classes=4)

                feature_input = Input(shape=(sequence_length, X_train_sequences.shape[2]))
                feature_inputs.append(feature_input)
        
        if part_fusion == "early":

            input_shape = X_train_sequences.shape[2]

            model = build_early_late_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg)

            num_classes = len(np.unique(y_train))
            print("Num classes: ", num_classes)
            print("Unique labels in y_train:", np.unique(y_train))
            print("Unique labels in y_val:", np.unique(y_val))
            print("Unique labels in y_test:", np.unique(y_test))

            model.add(Dense(dense_units, activation=activation))
            model.add(Dense(num_classes, activation="softmax"))
        
        else:
            feature_outputs = []

            for feature_input in feature_inputs:
                x = feature_input
                for _ in range(num_gru_layers):
                    x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                    x = Dropout(dropout)(x)
                    x = BatchNormalization()(x)
                feature_outputs.append(x)

            concatenated_features = concatenate(feature_outputs)

            x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(concatenated_features)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            
            model.add(Dense(dense_units, activation=activation))
            model.add(Dense(num_classes, activation="softmax"))

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
                f't{fold}_loss': model_history.history['loss'][epoch],
            }
            if 'accuracy' in model_history.history:
                metrics[f't{fold}_accuracy'] = model_history.history['accuracy'][epoch]
            if 'precision' in model_history.history:
                metrics[f't{fold}_precision'] = model_history.history['precision'][epoch]
            if 'recall' in model_history.history:
                metrics[f't{fold}_recall'] = model_history.history['recall'][epoch]
            if 'auc' in model_history.history:
                metrics[f't{fold}_auc'] = model_history.history['auc'][epoch]
            
            wandb.log(metrics)

        y_predict_probs = model.predict([X_test_sequences])

        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        print(test_metrics)

        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        
        #put "test" before metrics
        test_metrics = {f"t{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average test metrics across all folds:", avg_test_metrics)


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

    test_metrics_list = { "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }

    part_fusion = config.part_fusion

    for fold in range(5):

        if kernel_regularizer == "l1":
                reg = l1(0.01)
        elif kernel_regularizer == "l2":
            reg = l2(0.01)
        elif kernel_regularizer == "l1_l2":
            reg = l1_l2(0.01, 0.01)
        else:
            reg = None
        
        if part_fusion == "early":

            splits_pose, splits_facial, splits_audio = None, None, None

            if feature_set == "full":
                if data in ["reg", "norm"]:
                    splits_pose = create_data_splits(df_pose, "multiclass", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_facial = create_data_splits(df_facial, "multiclass", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits(df_audio, "multiclass", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
                elif data == "pca":
                    splits_pose = create_data_splits_pca(df_pose, "multiclass", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_facial = create_data_splits_pca(df_facial, "multiclass", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits_pca(df_audio, "multiclass", fold_no=0, num_folds=5, seed_value=42, sequence_length=sequence_length)

                if splits_pose is None: raise ValueError(f"Failed to create splits for pose modality.")
                if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
                if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")

            elif feature_set == "stats":
                if data in ["reg", "norm"]:
                    splits_facial = create_data_splits(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                elif data == "pca":
                    splits_facial = create_data_splits_pca(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits_pca(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

                if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
                if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")
            
            if feature_set == "full":
                X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio
            
            elif feature_set == "stats":
                X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio
            
            y_train_sequences = to_categorical(y_train_sequences, num_classes=4)
            y_val_sequences = to_categorical(y_val_sequences, num_classes=4)
            y_test_sequences = to_categorical(y_test_sequences, num_classes=4)

            if feature_set == "full":
                feature_inputs = [
                    Input(shape=(sequence_length, X_train_pose_seq.shape[2])),
                    Input(shape=(sequence_length, X_train_facial_seq.shape[2])),
                    Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                ]
                feature_outputs = []
            
            elif feature_set == "stats":
                feature_inputs = [
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

            num_classes = len(np.unique(y_train))
            print("Num classes: ", num_classes)
            print("Unique labels in y_train:", np.unique(y_train))
            print("Unique labels in y_val:", np.unique(y_val))
            print("Unique labels in y_test:", np.unique(y_test))
            x = Dense(dense_units, activation=activation)(x)
            x = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=feature_inputs, outputs=x)

        elif part_fusion == "intermediate":
            participant_pose_dfs = [group for _, group in df_pose.groupby('participant')]
            participant_facial_dfs = [group for _, group in df_facial.groupby('participant')]
            participant_audio_dfs = [group for _, group in df_audio.groupby('participant')]

            feature_inputs = []
            feature_outputs = []

            for participant_pose, participant_facial, participant_audio in zip(participant_pose_dfs, participant_facial_dfs, participant_audio_dfs):
                
                def process_modality(input_data):
                    x = input_data
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    return x
                
                processed = []
                
                if feature_set == "full":
                    if data in ["reg", "norm"]:
                        splits_pose = create_data_splits(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_pose = create_data_splits_pca(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_pose, X_val_pose, X_test_pose, _, _, _, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_pose = Input(shape=(sequence_length, X_train_pose_seq.shape[2]))
                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                    
                    processed_pose = process_modality(input_pose)
                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)
                    processed = [processed_pose, processed_facial, processed_audio]
                
                if feature_set == "stats":
                    if data in ["reg", "norm"]:
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                    
                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)
                    processed = [processed_facial, processed_audio]  

                participant_fused_features = concatenate([processed_pose, processed_facial, processed_audio])
                feature_outputs.append(participant_fused_features)

            all_participants_fused = concatenate(feature_outputs)

            x = all_participants_fused
            for _ in range(num_gru_layers):
                x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)

            x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)

            num_classes = len(np.unique(y_train_sequences))
            x = Dense(dense_units, activation=activation)(x)
            output = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=feature_inputs, outputs=output)

        elif part_fusion == "late":
            participant_pose_dfs = [group for _, group in df_pose.groupby('participant')]
            participant_facial_dfs = [group for _, group in df_facial.groupby('participant')]
            participant_audio_dfs = [group for _, group in df_audio.groupby('participant')]

            participant_outputs = []

            for participant_pose, participant_facial, participant_audio in zip(participant_pose_dfs, participant_facial_dfs, participant_audio_dfs):
                
                def process_modality(input_data):
                    x = input_data
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    return x
                
                processed = []
                
                if feature_set == "full":
                    if data in ["reg", "norm"]:
                        splits_pose = create_data_splits(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_pose = create_data_splits_pca(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_pose, X_val_pose, X_test_pose, _, _, _, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_pose = Input(shape=(sequence_length, X_train_pose_seq.shape[2]))
                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                    
                    processed_pose = process_modality(input_pose)
                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)
                    processed = [processed_pose, processed_facial, processed_audio]
                
                if feature_set == "stats":
                    if data in ["reg", "norm"]:
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                    
                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)
                    processed = [processed_facial, processed_audio]                                  

                participant_fused_features = concatenate(processed)

                x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(participant_fused_features)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)

                participant_output = Dense(dense_units, activation=activation)(x)
                participant_outputs.append(participant_output)

            final_fused_features = concatenate(participant_outputs)

            output = Dense(num_classes, activation="softmax")(final_fused_features)

            model = Model(inputs=[input_pose, input_facial, input_audio] * len(participant_pose_dfs), outputs=output)
                
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

        if feature_set == "full":
            model_history = model.fit(
                [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2
            )
        
        elif feature_set == "stats":
            model_history = model.fit(
                [X_train_facial_seq, X_train_audio_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_facial_seq, X_val_audio_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2
            )
        
        for epoch in range(len(model_history.history['loss'])):
            metrics = {
                f't{fold}_loss': model_history.history['loss'][epoch],
            }
            if 'accuracy' in model_history.history:
                metrics[f't{fold}_accuracy'] = model_history.history['accuracy'][epoch]
            if 'precision' in model_history.history:
                metrics[f't{fold}_precision'] = model_history.history['precision'][epoch]
            if 'recall' in model_history.history:
                metrics[f't{fold}_recall'] = model_history.history['recall'][epoch]
            if 'auc' in model_history.history:
                metrics[f't{fold}_auc'] = model_history.history['auc'][epoch]
            
            wandb.log(metrics)

        if feature_set == "full":
            y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
        elif feature_set == "stats":
            y_predict_probs = model.predict([X_test_facial_seq, X_test_audio_seq])
            
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        print(test_metrics)

        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        
        test_metrics = {f"t{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average test metrics across all folds:", avg_test_metrics)


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
    part_fusion = config.part_fusion

    test_metrics_list = { "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }

    for fold in range(5):

        if kernel_regularizer == "l1":
                reg = l1(0.01)
        elif kernel_regularizer == "l2":
            reg = l2(0.01)
        elif kernel_regularizer == "l1_l2":
            reg = l1_l2(0.01, 0.01)
        else:
            reg = None
        
        if part_fusion == "early":

            splits_pose, splits_facial, splits_audio = None, None, None

            if feature_set == "full":
                if data in ["reg", "norm"]:
                    splits_pose = create_data_splits(df_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_facial = create_data_splits(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                elif data == "pca":
                    splits_pose = create_data_splits_pca(df_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_facial = create_data_splits_pca(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits_pca(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

                if splits_pose is None: raise ValueError(f"Failed to create splits for pose modality.")
                if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
                if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")

            elif feature_set == "stats":
                if data in ["reg", "norm"]:
                    splits_facial = create_data_splits(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                elif data == "pca":
                    splits_facial = create_data_splits_pca(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    splits_audio = create_data_splits_pca(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

                if splits_facial is None: raise ValueError(f"Failed to create splits for facial modality.")
                if splits_audio is None: raise ValueError(f"Failed to create splits for audio modality.")
            
            if feature_set == "full":
                X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
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

            elif feature_set == "stats":
                X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio
                
                facial_input = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                audio_input = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))

                facial_input = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                audio_input = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))

                facial_model = build_early_late_model(sequence_length, X_train_facial_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
                audio_model = build_early_late_model(sequence_length, X_train_audio_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)

                facial_output = facial_model(facial_input)
                audio_output = audio_model(audio_input)
                concatenated = concatenate([facial_output, audio_output])

            num_classes = len(np.unique(y_train))
            print("Num classes: ", num_classes)
            print("Unique labels in y_train:", np.unique(y_train))
            print("Unique labels in y_val:", np.unique(y_val))
            print("Unique labels in y_test:", np.unique(y_test))

            x = Dense(dense_units, activation=activation)(concatenated)
            output = Dense(num_classes, activation="softmax")(x)

            if feature_set == "full":
                model = Model(inputs=[pose_input, facial_input, audio_input], outputs=output)
            elif feature_set == "stats":
                model = Model(inputs=[facial_input, audio_input], outputs=output)

        elif part_fusion == "intermediate":
            participant_pose_dfs = [group for _, group in df_pose.groupby('participant')]
            participant_facial_dfs = [group for _, group in df_facial.groupby('participant')]
            participant_audio_dfs = [group for _, group in df_audio.groupby('participant')]

            feature_inputs = []
            feature_outputs = []

            for participant_pose, participant_facial, participant_audio in zip(participant_pose_dfs, participant_facial_dfs, participant_audio_dfs):
                
                def process_modality(input_data):
                    x = input_data
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    return x

                processed = []

                if feature_set == "full":
                    if data in ["reg", "norm"]:
                        splits_pose = create_data_splits(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_pose = create_data_splits_pca(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_pose, X_val_pose, X_test_pose, _, _, _, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_pose = Input(shape=(sequence_length, X_train_pose_seq.shape[2]))
                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))

                    processed_pose = process_modality(input_pose)
                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)

                    processed.extend([processed_pose, processed_facial, processed_audio])
                
                elif feature_set == "stats":
                    if data in ["reg", "norm"]:
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))

                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)

                    processed.extend([processed_facial, processed_audio])

                participant_fused_features = concatenate(processed)
                feature_outputs.append(participant_fused_features)

            all_participants_fused = concatenate(feature_outputs)

            x = all_participants_fused
            for _ in range(num_gru_layers):
                x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)

            x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)

            num_classes = len(np.unique(y_train_sequences))
            x = Dense(dense_units, activation=activation)(x)
            output = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=feature_inputs, outputs=output)

        elif part_fusion == "late":

            participant_pose_dfs = [group for _, group in df_pose.groupby('participant')]
            participant_facial_dfs = [group for _, group in df_facial.groupby('participant')]
            participant_audio_dfs = [group for _, group in df_audio.groupby('participant')]

            feature_inputs = []
            participant_outputs = []

            for participant_pose, participant_facial, participant_audio in zip(participant_pose_dfs, participant_facial_dfs, participant_audio_dfs):

                def process_modality(input_data):
                    x = input_data
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    return x

                processed = []

                if feature_set == "full":
                    if data in ["reg", "norm"]:
                        splits_pose = create_data_splits(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_pose = create_data_splits_pca(participant_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_pose is None or splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_pose, X_val_pose, X_test_pose, _, _, _, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_pose = Input(shape=(sequence_length, X_train_pose_seq.shape[2]))
                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                    feature_inputs.extend([input_pose, input_facial, input_audio])

                    processed_pose = process_modality(input_pose)
                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)
                    processed.extend([processed_pose, processed_facial, processed_audio])

                elif feature_set == "stats":
                    if data in ["reg", "norm"]:
                        splits_facial = create_data_splits(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")
                    elif data == "pca":
                        splits_facial = create_data_splits_pca(participant_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        splits_audio = create_data_splits_pca(participant_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                        if splits_facial is None or splits_audio is None:
                            raise ValueError("Failed to create splits for one or more modalities.")

                    X_train_facial, X_val_facial, X_test_facial, _, _, _, X_train_facial_seq, _, X_val_facial_seq, _, X_test_facial_seq, _, _ = splits_facial
                    X_train_audio, X_val_audio, X_test_audio, _, _, _, X_train_audio_seq, _, X_val_audio_seq, _, X_test_audio_seq, _, _ = splits_audio

                    input_facial = Input(shape=(sequence_length, X_train_facial_seq.shape[2]))
                    input_audio = Input(shape=(sequence_length, X_train_audio_seq.shape[2]))
                    feature_inputs.extend([input_facial, input_audio])

                    processed_facial = process_modality(input_facial)
                    processed_audio = process_modality(input_audio)
                    processed.extend([processed_facial, processed_audio])

                participant_fused_features = concatenate(processed)
                participant_outputs.append(participant_fused_features)

            all_participants_fused = concatenate(participant_outputs)

            num_classes = len(np.unique(y_train_sequences))

            x = Dense(dense_units, activation=activation)(all_participants_fused)
            output = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=feature_inputs, outputs=output)

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

        if feature_set == "full":
            model_history = model.fit(
                [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2
            )
        
        elif feature_set == "stats":
            model_history = model.fit(
                [X_train_facial_seq, X_train_audio_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_facial_seq, X_val_audio_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2
            )
        
        for epoch in range(len(model_history.history['loss'])):
            metrics = {
                f't{fold}_loss': model_history.history['loss'][epoch],
            }
            if 'accuracy' in model_history.history:
                metrics[f't{fold}_accuracy'] = model_history.history['accuracy'][epoch]
            if 'precision' in model_history.history:
                metrics[f't{fold}_precision'] = model_history.history['precision'][epoch]
            if 'recall' in model_history.history:
                metrics[f't{fold}_recall'] = model_history.history['recall'][epoch]
            if 'auc' in model_history.history:
                metrics[f't{fold}_auc'] = model_history.history['auc'][epoch]
            
            wandb.log(metrics)

        y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])

        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        print(test_metrics)

        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        
        #put "test" before metrics
        test_metrics = {f"t{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average test metrics across all folds:", avg_test_metrics)


def train():

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    feature_set = config.feature_set
    modality_full = config.modality_full
    modality_stats = config.modality_stats
    modality_rf = config.modality_rf
    data = config.data
    feat_fusion = config.feat_fusion

    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3.csv")
    # rf features: gaze_0_x, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y

    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = df.iloc[:, 28:63]
    df_audio_index = df.iloc[:, 63:]

    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]

    if feature_set == "full":
        modality = modality_full
    elif feature_set == "stats":
        modality = modality_stats
    elif feature_set == "rf":
        modality = modality_rf

    print("--------\nfeature set: ", feature_set, "\nmodality: ", modality, "\ndata: ", data, "\nfusion: ", feat_fusion, "\n--------")

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
            df = modality_mapping_stats.get(modality)

        if data != "reg":
            df = create_normalized_df(df)

        return df

    if feature_set == "full":
        df = get_modality_data(modality_full, data)

        if feat_fusion == "early" and modality == "combined":
            train_early_fusion(df, config)
        elif feat_fusion == "intermediate" and modality == "combined":
            train_intermediate_fusion(
                get_modality_data("pose", data),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        elif feat_fusion == "late" and modality == "combined":
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

        if feat_fusion == "early" and modality == "combined":
            train_early_fusion(df, config)
        elif feat_fusion == "intermediate" and modality == "combined":
            train_intermediate_fusion(
                pd.DataFrame(),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        elif feat_fusion == "late" and modality == "combined":
            train_late_fusion(
                pd.DataFrame(),
                get_modality_data("facial", data),
                get_modality_data("audio", data),
                config,
            )
        else:
            train_single_modality_model(get_modality_data(modality, data), config)
    
    elif feature_set == "rf":
        df = df_rf
        train_early_fusion(df, config)

def main():
    
    sweep_config = {
        'method': 'random',
        'name': 'gru_multiclass_all_v2',
        'parameters': {
            'feature_set' : {'values': ["full", "stats" ]}, # "rf"
            'modality_full' : {'values': ['combined', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio']},
            'modality_stats' : {'values': ['combined', 'facial', 'audio']},
            'modality_rf' : {'values': ['combined', 'pose', 'audio']},
            'data' : {'values' : ["reg", "norm", "pca"]},
            'feat_fusion': {'values': ['early', 'intermediate', 'late']},
            'part_fusion': {'values': ['early', 'intermediate', 'late']},

            'use_bidirectional': {'values': [True, False]},
            'num_gru_layers': {'values': [1, 2, 3]},
            'gru_units': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'dense_units': {'values': [32, 64, 128]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'value': 5},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss' : {'values' : ["categorical_crossentropy"]},
            'sequence_length' : {'values' : [30, 60, 90]}
        }
        # feature set (full, stats, rf) -> modality selection (combined, pose, facial, etc.) -> (reg, norm, pca) -> fusion
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="gru_multiclass_all_v2")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
