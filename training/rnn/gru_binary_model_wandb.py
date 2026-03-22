import os
import wandb
import numpy as np
import pandas as pd
import random
import gc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2, l1, l2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from create_data_splits import create_data_splits, create_data_splits_pca
from get_all_metrics import get_all_metrics
from gru_binary_single_modality import train_single_modality_model


# Resolve absolute paths so the script works regardless of CWD (e.g. SLURM)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "preprocessing"))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))

# Feature sets that use the full dataset without modality selection
NO_MODALITY_SELECTION_SETS = {"catch22", "tsfresh", "curated_features_v5_100fps", "rf", "selectkbest"}


def apply_windowing(df, window_size, stride, aggregation):
    """Aggregate a per-frame DataFrame into windowed samples.

    Windowing is applied per participant (column 0). Features (columns 4+)
    are aggregated with the specified method. Labels follow:
      - 'average' / 'last'  → last frame of the window
      - 'mode'              → statistical mode of the window

    Returns
    -------
    windowed_df : pd.DataFrame
        New DataFrame with one row per window.
    last_positions : list[int]
        Positional indices (in the reset-index view of df) of the last frame
        of each window — used to align auxiliary DataFrames (e.g. text embeddings).
    """
    df_reset = df.reset_index(drop=True)
    participant_col = df_reset.columns[1]
    info_cols = df_reset.columns[:4].tolist()
    feature_cols = df_reset.columns[4:].tolist()
    binary_col = info_cols[2]
    multiclass_col = info_cols[3]

    rows = []
    last_positions = []

    for participant in df_reset[participant_col].unique():
        p_positions = np.where(df_reset[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            last_pos = int(p_positions[end - 1])
            window = df_reset.iloc[window_pos]
            last_row = df_reset.iloc[last_pos]
            last_positions.append(last_pos)

            row = {
                info_cols[0]: last_row[info_cols[0]],
                info_cols[1]: last_row[info_cols[1]],
            }

            # Labels
            if aggregation == "mode":
                row[binary_col] = window[binary_col].mode().iloc[0]
                row[multiclass_col] = window[multiclass_col].mode().iloc[0]
            else:  # 'average' or 'last'
                row[binary_col] = last_row[binary_col]
                row[multiclass_col] = last_row[multiclass_col]

            # Features
            if aggregation == "average":
                for col in feature_cols:
                    row[col] = window[col].mean()
            elif aggregation == "mode":
                for col in feature_cols:
                    row[col] = window[col].mode().iloc[0]
            else:  # 'last'
                for col in feature_cols:
                    row[col] = last_row[col]

            rows.append(row)
            start += stride

    windowed_df = pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)
    windowed_df = windowed_df.dropna(axis=1, how="any").dropna(axis=0, how="any").reset_index(drop=True)

    print(f"Applied windowing: window_size={window_size}, stride={stride}, aggregation={aggregation}")
    print(f"Original samples: {len(df)}, Windowed samples: {len(windowed_df)}")
    return windowed_df, last_positions


def apply_catch22_windowing(df, window_size, stride):
    """Apply catch22 feature extraction on sliding windows per participant.

    For each window, catch22 computes 22 time-series features per input
    feature column (e.g. col__DN_HistogramMode_5, ...).
    Labels use the last frame of each window.

    Returns
    -------
    pd.DataFrame with 4 info columns + (n_features * 22) catch22 columns.
    """
    import pycatch22

    df_reset = df.reset_index(drop=True)
    participant_col = df_reset.columns[1]
    info_cols = df_reset.columns[:4].tolist()
    feature_cols = df_reset.columns[4:].tolist()
    binary_col = info_cols[2]
    multiclass_col = info_cols[3]

    rows = []
    for participant in df_reset[participant_col].unique():
        p_positions = np.where(df_reset[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            window = df_reset.iloc[window_pos]
            last_row = df_reset.iloc[int(p_positions[end - 1])]

            row = {
                info_cols[0]: last_row[info_cols[0]],
                info_cols[1]: last_row[info_cols[1]],
                binary_col: last_row[binary_col],
                multiclass_col: last_row[multiclass_col],
            }

            for col in feature_cols:
                series = window[col].values.tolist()
                result = pycatch22.catch22_all(series)
                for name, val in zip(result["names"], result["values"]):
                    row[f"{col}__{name}"] = val

            rows.append(row)
            start += stride

    result = pd.DataFrame(rows).reset_index(drop=True)
    result = result.dropna(axis=1, how="any").dropna(axis=0, how="any").reset_index(drop=True)
    return result


def apply_tsfresh_windowing(df, window_size, stride):
    """Apply tsfresh feature extraction on sliding windows per participant.

    Constructs a long-format DataFrame for all windows, then calls
    tsfresh.extract_features() in a single pass for efficiency.
    Labels use the last frame of each window. NaN feature columns are dropped.

    Returns
    -------
    pd.DataFrame with 4 info columns + tsfresh-extracted feature columns.
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters

    df_reset = df.reset_index(drop=True)
    participant_col = df_reset.columns[1]
    info_cols = df_reset.columns[:4].tolist()
    feature_cols = df_reset.columns[4:].tolist()
    binary_col = info_cols[2]
    multiclass_col = info_cols[3]

    info_rows = []
    window_slices = []
    window_id = 0

    for participant in df_reset[participant_col].unique():
        p_positions = np.where(df_reset[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            window = df_reset.iloc[window_pos]
            last_row = df_reset.iloc[int(p_positions[end - 1])]

            info_rows.append({
                info_cols[0]: last_row[info_cols[0]],
                info_cols[1]: last_row[info_cols[1]],
                binary_col: last_row[binary_col],
                multiclass_col: last_row[multiclass_col],
                "_window_id": window_id,
            })

            slice_df = window[feature_cols].copy()
            slice_df = slice_df.reset_index(drop=True)
            slice_df["_window_id"] = window_id
            slice_df["_time"] = range(window_size)
            window_slices.append(slice_df)

            window_id += 1
            start += stride

    tsfresh_input = pd.concat(window_slices, ignore_index=True)
    extracted = extract_features(
        tsfresh_input,
        column_id="_window_id",
        column_sort="_time",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True,
    )
    # Drop columns with any NaN (some tsfresh features fail on short windows)
    extracted = extracted.dropna(axis=1, how="any")
    extracted = extracted.reset_index(drop=True)

    info_df = pd.DataFrame(info_rows).drop(columns=["_window_id"]).reset_index(drop=True)
    result = pd.concat([info_df, extracted], axis=1)
    return result.dropna(axis=0, how="any").reset_index(drop=True)


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

    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": [],
        "test_auc": [],
        "test_fnr": [],
        "test_windowed_accuracy": [],
        "test_windowed_precision": [],
        "test_windowed_recall": [],
        "test_windowed_f1": [],
        "test_earliest_detection_time": [],
    }

    for fold in range(5):

        print("Fold ", fold)

        splits = create_data_splits(
                df, "binary",
                fold_no=fold,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
        if splits is None:
            return

        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length, session_train, session_val, session_test = splits

        print("X_train_sequences shape:", X_train_sequences.shape)
        print("X_val_sequences shape:", X_val_sequences.shape)
        print("X_test_sequences shape:", X_test_sequences.shape)
        print("y_train_sequences shape:", y_train_sequences.shape)
        print("y_val_sequences shape:", y_val_sequences.shape)
        print("y_test_sequences shape:", y_test_sequences.shape)


        if kernel_regularizer == "l1":
            reg = l1(0.01)
        elif kernel_regularizer == "l2":
            reg = l2(0.01)
        elif kernel_regularizer == "l1_l2":
            reg = l1_l2(l1=0.01, l2=0.01)
        else:
            reg = None
        
        input_shape = X_train_sequences.shape[2]

        model = build_early_late_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg)
        
        num_classes = len(np.unique(y_train))
        print("Num classes: ", num_classes)
        print("Unique labels in y_train:", np.unique(y_train))
        print("Unique labels in y_val:", np.unique(y_val))
        print("Unique labels in y_test:", np.unique(y_test))

        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        if optimizer == 'adam':
            optim = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optim = optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optim = optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(optimizer=optim, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

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
                    'fold': fold,
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
        y_predict_probs_clean = np.nan_to_num(y_predict_probs, nan=0.0)

        df_probs = pd.DataFrame(y_predict_probs_clean)

        y_pred = (y_predict_probs_clean > 0.5).astype(int).flatten()

        if len(y_test_sequences.shape) > 1 and y_test_sequences.shape[1] > 1:
            y_test_class_indices = np.argmax(y_test_sequences, axis=1)
        else:
            y_test_class_indices = y_test_sequences

        y_test = y_test_class_indices

        cm = confusion_matrix(y_test, y_pred)

        unique, counts = np.unique(y_test, return_counts=True)
        print("\nTest label distribution:")
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count}")

        print("\nConfusion Matrix (Binary):")
        print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        wandb.log({f"fold_{fold}_confusion_matrix": cm})

        effective_test_stride = min(config.test_stride, config.test_window_size)
        wandb.log({"effective_test_stride": effective_test_stride, "effective_test_window_size": config.test_window_size})
        test_metrics = get_all_metrics(y_pred, y_test_class_indices, y_pred_proba=y_predict_probs_clean,
                                       sessions=session_test[-len(y_pred):], window_size=config.test_window_size,
                                       stride=effective_test_stride, tolerance=1)
        test_metrics.pop("test_edt_per_session", None)

        for key in test_metrics:
            if key in test_metrics_list:
                test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)

        tf.keras.backend.clear_session()
        gc.collect()
    
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def train_intermediate_fusion(modality_dfs, config):

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

    modality_keys = list(modality_dfs.keys())

    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": [],
        "test_auc": [],
        "test_fnr": [],
        "test_windowed_accuracy": [],
        "test_windowed_precision": [],
        "test_windowed_recall": [],
        "test_windowed_f1": [],
        "test_earliest_detection_time": [],
    }

    if kernel_regularizer == "l1":
        reg = l1(0.01)
    elif kernel_regularizer == "l2":
        reg = l2(0.01)
    elif kernel_regularizer == "l1_l2":
        reg = l1_l2(l1=0.01,l2=0.01)
    else:
        reg = None

    for fold in range(5):
        print("Fold ", fold)
        
        splits = {}
        for modality_key in modality_keys:
            df = modality_dfs[modality_key]
            splits[modality_key] = create_data_splits(df, "binary", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

        # Extract session info for get_all_metrics from first modality
        first_modality_splits = splits[modality_keys[0]]
        session_test = first_modality_splits[15]  # session_test
        
        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7] 
        y_val_sequences = splits[first_modality][9] 
        y_test_sequences = splits[first_modality][11] 
        
        feature_inputs = []
        feature_outputs = []
        
        for modality_key in modality_keys:
            X_train_seq = splits[modality_key][6]
            
            feature_input = Input(shape=(sequence_length, X_train_seq.shape[2]))
            feature_inputs.append(feature_input)
            
            x = feature_input
            for _ in range(num_gru_layers):
                if use_bidirectional:
                    x = Bidirectional(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg))(x)
                else:
                    x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)
            feature_outputs.append(x)
        
        concatenated_features = concatenate(feature_outputs)
        
        x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(concatenated_features)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)

        x = Dense(dense_units, activation=activation)(x)
        x = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=feature_inputs, outputs=x)
        model.summary()

        if optimizer == 'adam':
            optim = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optim = optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optim = optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optim, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
        
        train_inputs = [splits[m][6] for m in modality_keys]
        val_inputs = [splits[m][8] for m in modality_keys] 
        test_inputs = [splits[m][10] for m in modality_keys]
        
        model_history = model.fit(
            train_inputs, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val_sequences),
            verbose=2
        )
        
        for epoch in range(len(model_history.history['loss'])):

            metrics = {
                'fold': fold,
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
        
        y_predict_probs = model.predict(test_inputs)
        y_predict_probs_clean = np.nan_to_num(y_predict_probs, nan=0.0)
        
        df_probs = pd.DataFrame(y_predict_probs_clean)
        
        y_pred = (y_predict_probs_clean > 0.5).astype(int).flatten()

        if len(y_test_sequences.shape) > 1 and y_test_sequences.shape[1] > 1:
            y_test_class_indices = np.argmax(y_test_sequences, axis=1)
        else:
            y_test_class_indices = y_test_sequences

        y_test = y_test_class_indices

        cm = confusion_matrix(y_test, y_pred)

        unique, counts = np.unique(y_test, return_counts=True)
        print("\nTest label distribution:")
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count}")

        print("\nConfusion Matrix (Binary):")
        print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        wandb.log({f"fold_{fold}_confusion_matrix": cm})

        effective_test_stride = min(config.test_stride, config.test_window_size)
        wandb.log({"effective_test_stride": effective_test_stride, "effective_test_window_size": config.test_window_size})
        test_metrics = get_all_metrics(y_pred, y_test_class_indices, y_pred_proba=y_predict_probs_clean,
                                       sessions=session_test[-len(y_pred):], window_size=config.test_window_size,
                                       stride=effective_test_stride, tolerance=1)
        test_metrics.pop("test_edt_per_session", None)
        
        for key in test_metrics:
            if key in test_metrics_list:
                test_metrics_list[key].append(test_metrics[key])
        
        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)

        tf.keras.backend.clear_session()
        gc.collect()

    
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def train_late_fusion(modality_dfs, config):

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

    modality_keys = list(modality_dfs.keys())

    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": [],
        "test_auc": [],
        "test_fnr": [],
        "test_windowed_accuracy": [],
        "test_windowed_precision": [],
        "test_windowed_recall": [],
        "test_windowed_f1": [],
        "test_earliest_detection_time": [],
    }

    if kernel_regularizer == "l1":
        reg = l1(0.01)
    elif kernel_regularizer == "l2":
        reg = l2(0.01)
    elif kernel_regularizer == "l1_l2":
        reg = l1_l2(l1=0.01,l2=0.01)
    else:
        reg = None

    for fold in range(5):
        print("Fold ", fold)
        
        splits = {}

        for modality_key in modality_keys:
            df = modality_dfs[modality_key]
            splits[modality_key] = create_data_splits(df, "binary", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

        # Extract session info for get_all_metrics from first modality
        first_modality_splits = splits[modality_keys[0]]
        session_test = first_modality_splits[15]  # session_test

        input_layers = []
        outputs = []

        for modality_key in modality_keys:
            X_train_seq = splits[modality_key][6]
            
            input_layer = Input(shape=(sequence_length, X_train_seq.shape[2]))
            
            model = build_early_late_model(
                sequence_length, 
                X_train_seq.shape[2], 
                num_gru_layers, 
                gru_units, 
                activation, 
                use_bidirectional, 
                dropout, 
                reg
            )
            
            input_layers.append(input_layer)
            outputs.append(model(input_layer))
        
        if len(outputs) > 1:
            concatenated = concatenate(outputs)
        else:
            concatenated = outputs[0]
        
        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7]
        y_val_sequences = splits[first_modality][9]
        y_test_sequences = splits[first_modality][11]

        x = Dense(dense_units, activation=activation)(concatenated)
        output_layer = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=input_layers, outputs=output_layer)

        model.summary()

        if optimizer == 'adam':
            optim = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optim = optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optim = optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optim, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
        
        train_inputs = [splits[m][6] for m in modality_keys]
        val_inputs = [splits[m][8] for m in modality_keys]  
        test_inputs = [splits[m][10] for m in modality_keys]
        
        model_history = model.fit(
            train_inputs, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val_sequences),
            verbose=2
        )
        
        for epoch in range(len(model_history.history['loss'])):

            metrics = {
                'fold': fold,
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

        y_predict_probs = model.predict(test_inputs)
        y_predict_probs_clean = np.nan_to_num(y_predict_probs, nan=0.0)

        df_probs = pd.DataFrame(y_predict_probs_clean)

        y_pred = (y_predict_probs_clean > 0.5).astype(int).flatten()

        if len(y_test_sequences.shape) > 1 and y_test_sequences.shape[1] > 1:
            y_test_class_indices = np.argmax(y_test_sequences, axis=1)
        else:
            y_test_class_indices = y_test_sequences

        y_test = y_test_class_indices

        cm = confusion_matrix(y_test, y_pred)

        unique, counts = np.unique(y_test, return_counts=True)
        print("\nTest label distribution:")
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count}")

        print("\nConfusion Matrix (Binary):")
        print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        wandb.log({f"fold_{fold}_confusion_matrix": cm})

        effective_test_stride = min(config.test_stride, config.test_window_size)
        wandb.log({"effective_test_stride": effective_test_stride, "effective_test_window_size": config.test_window_size})
        test_metrics = get_all_metrics(y_pred, y_test_class_indices, y_pred_proba=y_predict_probs_clean,
                                       sessions=session_test[-len(y_pred):], window_size=config.test_window_size,
                                       stride=effective_test_stride, tolerance=1)
        test_metrics.pop("test_edt_per_session", None)

        for key in test_metrics:
            if key in test_metrics_list:
                test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)

        tf.keras.backend.clear_session()
        gc.collect()


    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def validate_modality_feature_combination(modality, feature_set):
    modality_components = modality.split('_')
    
    # Text only works with 'full' feature set
    if 'text' in modality_components and feature_set != 'full':
        return False
    
    return True

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

def create_norm_pca_df(df):
    if df.empty:
        raise ValueError("create_norm_pca_df: Input DataFrame is empty.")
    participant_frames_labels = df.iloc[:, :4]

    x = df.iloc[:, 4:]
    x = StandardScaler().fit_transform(x.values)

    pca = PCA(n_components=0.90)
    principal_components = pca.fit_transform(x)
    print(principal_components.shape)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
    principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

    return principal_df

def train():

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

    feature_set = config.feature_set

    # Skip modality validation for feature sets that ignore modality selection
    if feature_set not in NO_MODALITY_SELECTION_SETS:
        is_valid_combination = validate_modality_feature_combination(config.modality, config.feature_set)
        if not is_valid_combination:
            print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
            wandb.log({"status": "skipped_invalid_combination"})
            return

    # ------------------------------------------------------------------
    # Load base 100fps dataset
    # ------------------------------------------------------------------
    if feature_set in ["curated_features_v5_100fps", "tsfresh", "catch22"]:
        df_base = pd.read_csv(os.path.join(DATA_DIR, "interpolated", "curated_features_v5_100fps.csv"))
    elif feature_set == "rf":
        df_base = pd.read_csv(os.path.join(DATA_DIR, "feature_sets", "rf_allparticipants_100fps.csv"))
    elif feature_set == "selectkbest":
        df_base = pd.read_csv(os.path.join(DATA_DIR, "feature_sets", "select_k_best_allparticipants_100fps_binary_label.csv"))
    else:
        # 'full' — all start from the full 100fps feature set
        df_base = pd.read_csv(os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv"))
        print("Unique participants in base dataset:", df_base.iloc[:, 1].unique())

    # ------------------------------------------------------------------
    # Apply windowing / on-the-fly feature extraction
    # ------------------------------------------------------------------
    window = config.window
    win_stride = min(config.stride, window)
    wandb.log({"effective_stride": win_stride, "effective_window": window})

    if feature_set == "catch22":
        df = apply_catch22_windowing(df_base, window, win_stride)
        last_positions = None
    elif feature_set == "tsfresh":
        df = apply_tsfresh_windowing(df_base, window, win_stride)
        last_positions = None
    else:
        df, last_positions = apply_windowing(df_base, window, win_stride, config.aggregation)

    # Skip text/cosine modality when last_positions is unavailable (catch22/tsfresh)
    modality_components = config.modality.split('_')
    if last_positions is None and ("text" in modality_components or "cosine" in modality_components):
        print(f"Skipping: text/cosine modality not supported with {feature_set} (no last_positions)")
        wandb.log({"status": "skipped_invalid_combination"})
        return

    # ------------------------------------------------------------------
    # Feature / modality selection (only for 'full' and 'selectkbest' feature sets)
    # rf is all facial features
    # ------------------------------------------------------------------
    data = config.dataset
    fusion_type = config.fusion_type

    if feature_set not in NO_MODALITY_SELECTION_SETS:
        info = df.iloc[:, :4]

        # Index-based modality splitting for 'full' feature set
        df_pose_index = df.iloc[:, 4:28]
        df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
        df_audio_index = df.iloc[:, 63:88]

        modality_components = config.modality.split('_')

        # Text embeddings must be aligned to the windowed rows via last_positions
        if "text" in modality_components or "cosine" in modality_components:
            df_text_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_embeddings_pca.csv"))
            df_text_cos_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_cosine_similarity.csv"))
            df_text_index = df_text_raw.iloc[last_positions, 4:].reset_index(drop=True)
            df_text_distance = df_text_cos_raw.iloc[last_positions, 4:].reset_index(drop=True)

        selected_modalities = {}
        if "pose" in modality_components:
            selected_modalities["pose_full"] = df_pose_index
        if "facial" in modality_components:
            selected_modalities["facial_full"] = df_facial_index
        if "audio" in modality_components:
            selected_modalities["audio_full"] = df_audio_index
        if "text" in modality_components:
            selected_modalities["text_full"] = df_text_index
        if "cosine" in modality_components:
            selected_modalities["text_distance"] = df_text_distance

        if fusion_type == "early":
            df = info
            for m in selected_modalities.values():
                df = pd.concat([df, m.reset_index(drop=True)], axis=1)

            if data == "norm":
                df = create_normalized_df(df)
            elif data == "pca":
                df = create_norm_pca_df(create_normalized_df(df))

            print(df)
            print(df.shape)

            all_features = df.columns[4:].tolist()
            wandb.log({"features_included": all_features, "n_features_selected": len(all_features), "total_available_features": len(all_features)})

            train_early_fusion(df, config)

        if fusion_type == "intermediate" or fusion_type == "late":
            dfs = {}

            if data == "norm":
                for modality_name, m in selected_modalities.items():
                    df_temp = pd.concat([info.copy(), m], axis=1)
                    dfs[modality_name] = create_normalized_df(df_temp)
            elif data == "pca":
                for modality_name, m in selected_modalities.items():
                    if modality_name == "text_full":
                        df_temp = pd.concat([info.copy(), m], axis=1)
                        dfs[modality_name] = df_temp
                    else:
                        df_temp = pd.concat([info.copy(), m], axis=1)
                        dfs[modality_name] = create_norm_pca_df(create_normalized_df(df_temp))
            elif data == "reg":
                for modality_name, m in selected_modalities.items():
                    df_temp = pd.concat([info.copy(), m], axis=1)
                    dfs[modality_name] = create_normalized_df(df_temp)

            print(dfs)

            if fusion_type == "intermediate":
                train_intermediate_fusion(dfs, config)
            elif fusion_type == "late":
                train_late_fusion(dfs, config)
    else:
        # curated_features_v5_100fps / catch22 / tsfresh / rf feature sets

        # Check if text/cosine should be appended (e.g. rf + text)
        modality_components = config.modality.split('_')
        has_text_modality = (
            ("text" in modality_components or "cosine" in modality_components)
            and last_positions is not None
        )

        # For early fusion, concatenate text features BEFORE normalization
        if has_text_modality and fusion_type == "early":
            if "text" in modality_components:
                df_text_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_embeddings_pca.csv"))
                df_text_index = df_text_raw.iloc[last_positions, 4:].reset_index(drop=True)
                df = pd.concat([df, df_text_index], axis=1)
            if "cosine" in modality_components:
                df_text_cos_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_cosine_similarity.csv"))
                df_text_distance = df_text_cos_raw.iloc[last_positions, 4:].reset_index(drop=True)
                df = pd.concat([df, df_text_distance], axis=1)

        if data == "norm":
            df = create_normalized_df(df)
        elif data == "pca":
            df = create_norm_pca_df(create_normalized_df(df))

        print(df)
        print(df.shape)

        # -------------------------------------------------------
        # Feature randomization (if enabled)
        # -------------------------------------------------------
        # do this only for curated dataset
        if config.feature_randomizer == 1 and config.feature_set == "curated_features_v5_100fps":
            available_features = df.columns[4:].tolist()
            max_features = len(available_features)

            if max_features > 0:
                n_features = np.random.randint(1, max_features + 1)
                selected_features = np.random.choice(
                    available_features,
                    size=n_features,
                    replace=False
                ).tolist()
                df = df[df.columns[:4].tolist() + selected_features]
                wandb.log({
                    "features_included": selected_features,
                    "n_features_selected": n_features,
                    "total_available_features": max_features
                })
                print(f"Feature randomization enabled: Selected {n_features} out of {max_features} features")
                print(f"Selected features: {selected_features}")
            else:
                print("Warning: No features available for randomization")
                wandb.log({"features_included": [], "n_features_selected": 0, "total_available_features": 0})
        else:
            all_features = df.columns[4:].tolist()
            wandb.log({
                "features_included": all_features,
                "n_features_selected": len(all_features),
                "total_available_features": len(all_features)
            })

        # Intermediate / late fusion with text modalities
        if has_text_modality and fusion_type in ("intermediate", "late"):
            info = df.iloc[:, :4]
            dfs = {"base_features": df}
            if "text" in modality_components:
                df_text_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_embeddings_pca.csv"))
                df_text_index = df_text_raw.iloc[last_positions, 4:].reset_index(drop=True)
                df_text = pd.concat([info.reset_index(drop=True), df_text_index], axis=1)
                if data == "norm":
                    df_text = create_normalized_df(df_text)
                elif data == "pca":
                    df_text = create_norm_pca_df(create_normalized_df(df_text))
                dfs["text_full"] = df_text
            if "cosine" in modality_components:
                df_text_cos_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_cosine_similarity.csv"))
                df_text_distance = df_text_cos_raw.iloc[last_positions, 4:].reset_index(drop=True)
                df_cos = pd.concat([info.reset_index(drop=True), df_text_distance], axis=1)
                if data == "norm":
                    df_cos = create_normalized_df(df_cos)
                elif data == "pca":
                    df_cos = create_norm_pca_df(create_normalized_df(df_cos))
                dfs["text_distance"] = df_cos

            print(dfs)
            if fusion_type == "intermediate":
                train_intermediate_fusion(dfs, config)
            elif fusion_type == "late":
                train_late_fusion(dfs, config)
        else:
            print(df)
            print(df.shape)
            train_early_fusion(df, config)


def main():

    sweep_config = {
        'method': 'random',
        'name': 'brirl_gru_inter',
        'parameters': {
            'feature_set': {'values': ['full', 'curated_features_v5_100fps', 'catch22', 'tsfresh', 'rf', 'selectkbest']},
            'modality': {'values': [
                'pose', 'facial', 'audio', 'text', 'cosine',
                'pose_facial', 'pose_audio', 'pose_text', 'pose_cosine',
                'facial_audio', 'facial_text', 'facial_cosine',
                'audio_text', 'audio_cosine',
                'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
                'facial_audio_text', 'facial_audio_cosine', 'pose_facial_audio_cosine',
                'pose_facial_audio_text', 'pose_audio_cosine',
            ]},

            'dataset': {'values': ["reg", "norm", "pca"]},
            'fusion_type': {'values': ['early', 'intermediate', 'late']},
            'feature_randomizer': {'values': [1]},

            # Dataset-level windowing (applied to 100fps data before training)
            'window': {'values': [1, 5, 10, 15, 30]},
            'stride': {'values': [1, 3, 5, 10]},
            'aggregation': {'values': ['average', 'mode', 'last']},
            # Test-time windowed metrics parameters
            'test_window_size': {'values': [1, 5, 10, 15, 30]},
            'test_stride': {'values': [1, 3, 5, 10]},

            'use_bidirectional': {'values': [True, False]},
            'num_gru_layers': {'values': [1, 2, 3]},
            'gru_units': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'dense_units': {'values': [32, 64, 128]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'values': [100, 150, 200, 250]},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss': {'values': ["binary_crossentropy"]},
            
            'sequence_length': {'values': [5, 10, 15, 30, 60, 100, 150, 300]},

            'model': {'values': ['gru']},
            'class': {'values': ['binary']}
            
        }
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_inter")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
