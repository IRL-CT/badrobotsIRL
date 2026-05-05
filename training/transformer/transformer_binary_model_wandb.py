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
from keras.models import Model
from keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    LayerNormalization, MultiHeadAttention, Add,
    GlobalAveragePooling1D, concatenate,
)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1_l2, l1, l2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from create_data_splits import create_data_splits, create_data_splits_pca
from get_all_metrics import get_all_metrics


# Prevent TF from pre-allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "preprocessing"))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))

# Feature sets that use the full dataset without modality selection
NO_MODALITY_SELECTION_SETS = {"catch22", "tsfresh", "curated_features_v5_100fps", "rf", "selectkbest"}


# Windowing helpers (from rnn scripts)

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
    """Apply catch22 feature extraction on sliding windows per participant."""
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
    """Apply tsfresh feature extraction on sliding windows per participant."""
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
    extracted = extracted.dropna(axis=1, how="any")
    extracted = extracted.reset_index(drop=True)

    info_df = pd.DataFrame(info_rows).drop(columns=["_window_id"]).reset_index(drop=True)
    result = pd.concat([info_df, extracted], axis=1)
    return result.dropna(axis=0, how="any").reset_index(drop=True)


# Transformer building blocks

class CLSTokenPrepend(tf.keras.layers.Layer):
    """Prepend a learnable [CLS] token to every sequence."""

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        return tf.concat([cls_tokens, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


class LearnablePositionalEncoding(tf.keras.layers.Layer):
    """Add learnable positional embeddings to the input."""

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.max_len, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class ExtractCLSToken(tf.keras.layers.Layer):
    """Extract the [CLS] token (position 0) from a sequence."""

    def call(self, x):
        return x[:, 0, :]


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Single Pre-LN Transformer encoder block.

    Architecture:
        x → LayerNorm → MultiHeadAttention → Dropout → Add(x) →
        → LayerNorm → FFN → Dropout → Add(x)
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att_ln = LayerNormalization(epsilon=1e-6)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.att_dropout = Dropout(dropout_rate)

        self.ffn_ln = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(dropout_rate),
            Dense(d_model),
        ])
        self.ffn_dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        # Pre-LN attention
        x_norm = self.att_ln(x)
        attn_output = self.att(x_norm, x_norm, training=training)
        attn_output = self.att_dropout(attn_output, training=training)
        x = x + attn_output

        # Pre-LN FFN
        x_norm = self.ffn_ln(x)
        ffn_output = self.ffn(x_norm, training=training)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        x = x + ffn_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def build_transformer_encoder(sequence_length, input_dim, d_model, num_heads,
                               num_encoder_layers, ff_dim, dropout_rate, name="transformer_encoder"):
    """Build a Transformer encoder with learnable positional encoding and CLS token.

    Returns
    -------
    keras.Model
        Model mapping (batch, sequence_length, input_dim) → (batch, d_model).
        Output is the [CLS] token representation.
    """
    inputs = Input(shape=(sequence_length, input_dim))

    # Project input features to d_model
    x = Dense(d_model)(inputs)

    # Prepend CLS token: (batch, seq_len+1, d_model)
    x = CLSTokenPrepend(d_model)(x)

    # Positional embedding for seq_len+1 positions
    x = LearnablePositionalEncoding(sequence_length + 1, d_model)(x)

    x = Dropout(dropout_rate)(x)

    # Stacked encoder blocks
    for i in range(num_encoder_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"encoder_block_{i}",
        )(x)

    # Final layer norm
    x = LayerNormalization(epsilon=1e-6)(x)

    # Extract CLS token output (position 0)
    cls_output = ExtractCLSToken()(x)

    return Model(inputs=inputs, outputs=cls_output, name=name)


# Training: early, intermediate, late

def train_early_fusion(df, config):

    d_model = config.d_model
    num_heads = config.num_heads
    num_encoder_layers = config.num_encoder_layers
    ff_dim = config.ff_dim
    batch_size = config.batch_size
    epochs = config.epochs
    dropout = config.dropout_rate
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
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

        input_shape = X_train_sequences.shape[2]

        # Build transformer encoder
        encoder = build_transformer_encoder(
            sequence_length=sequence_length,
            input_dim=input_shape,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            ff_dim=ff_dim,
            dropout_rate=dropout,
        )

        # Classification head
        inputs = Input(shape=(sequence_length, input_shape))
        x = encoder(inputs)
        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout)(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)

        num_classes = len(np.unique(y_train))
        print("Num classes: ", num_classes)
        print("Unique labels in y_train:", np.unique(y_train))
        print("Unique labels in y_val:", np.unique(y_val))
        print("Unique labels in y_test:", np.unique(y_test))

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

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        model_history = model.fit(
            X_train_sequences, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_sequences, y_val_sequences),
            callbacks=[early_stop],
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

        del model, encoder, model_history, optim, early_stop
        del X_train_sequences, y_train_sequences
        del X_val_sequences, y_val_sequences
        del X_test_sequences, y_test_sequences
        del X_train, X_val, X_test
        del y_predict_probs, y_predict_probs_clean
        tf.keras.backend.clear_session()
        gc.collect()

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def train_intermediate_fusion(modality_dfs, config):

    d_model = config.d_model
    num_heads = config.num_heads
    num_encoder_layers = config.num_encoder_layers
    ff_dim = config.ff_dim
    batch_size = config.batch_size
    epochs = config.epochs
    dropout = config.dropout_rate
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
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

        # Per-modality transformer encoders producing sequence outputs
        feature_inputs = []
        feature_outputs = []

        for modality_key in modality_keys:
            X_train_seq = splits[modality_key][6]
            input_dim = X_train_seq.shape[2]

            feature_input = Input(shape=(sequence_length, input_dim))
            feature_inputs.append(feature_input)

            # Project to d_model
            x = Dense(d_model)(feature_input)

            # Positional encoding
            x = LearnablePositionalEncoding(sequence_length, d_model)(x)
            x = Dropout(dropout)(x)

            # Per-modality encoder blocks (output full sequence, not CLS)
            for i in range(num_encoder_layers):
                x = TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout_rate=dropout,
                    name=f"enc_{modality_key}_{i}",
                )(x)

            feature_outputs.append(x)

        # Concatenate modality sequences along feature dim → shared encoder
        concatenated_features = concatenate(feature_outputs)  # (batch, seq_len, d_model * n_modalities)

        # Shared transformer encoder layer on concatenated features
        shared_d_model = d_model * len(modality_keys)
        x = TransformerEncoderBlock(
            d_model=shared_d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout,
            name="shared_encoder",
        )(concatenated_features)

        # Global average pooling over sequence dimension
        x = GlobalAveragePooling1D()(x)
        x = LayerNormalization(epsilon=1e-6)(x)

        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout)(x)
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

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        model_history = model.fit(
            train_inputs, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val_sequences),
            callbacks=[early_stop],
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

        del model, model_history, optim, early_stop
        del splits, train_inputs, val_inputs, test_inputs
        del feature_inputs, feature_outputs
        del y_train_sequences, y_val_sequences, y_test_sequences
        del y_predict_probs, y_predict_probs_clean
        tf.keras.backend.clear_session()
        gc.collect()

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def train_late_fusion(modality_dfs, config):

    d_model = config.d_model
    num_heads = config.num_heads
    num_encoder_layers = config.num_encoder_layers
    ff_dim = config.ff_dim
    batch_size = config.batch_size
    epochs = config.epochs
    dropout = config.dropout_rate
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
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
            input_dim = X_train_seq.shape[2]

            input_layer = Input(shape=(sequence_length, input_dim))

            # Build per-modality transformer encoder (outputs CLS token vector)
            encoder = build_transformer_encoder(
                sequence_length=sequence_length,
                input_dim=input_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                ff_dim=ff_dim,
                dropout_rate=dropout,
                name=f"transformer_encoder_{modality_key}",
            )

            input_layers.append(input_layer)
            outputs.append(encoder(input_layer))

        if len(outputs) > 1:
            concatenated = concatenate(outputs)
        else:
            concatenated = outputs[0]

        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7]
        y_val_sequences = splits[first_modality][9]
        y_test_sequences = splits[first_modality][11]

        x = Dense(dense_units, activation="relu")(concatenated)
        x = Dropout(dropout)(x)
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

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        model_history = model.fit(
            train_inputs, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val_sequences),
            callbacks=[early_stop],
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

        del model, model_history, optim, early_stop
        del splits, train_inputs, val_inputs, test_inputs
        del input_layers, outputs
        del y_train_sequences, y_val_sequences, y_test_sequences
        del y_predict_probs, y_predict_probs_clean
        tf.keras.backend.clear_session()
        gc.collect()

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def validate_modality_feature_combination(modality, feature_set):
    modality_components = modality.split('_')
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

    # Load dataset
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

    # Apply windowing / on-the-fly feature extraction
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

    # Free the raw dataframe after windowing
    del df_base
    gc.collect()

    # Memory guard: skip hyperparameter combinations that would exceed RAM
    n_samples = len(df)
    n_features = len(df.columns) - 4
    n_modalities = len(config.modality.split('_'))
    estimated_seq_bytes = n_samples * config.sequence_length * n_features * 4 * 3  # train/val/test
    if config.fusion_type in ('intermediate', 'late') and config.feature_set not in NO_MODALITY_SELECTION_SETS:
        estimated_seq_bytes *= n_modalities
    estimated_gb = estimated_seq_bytes / (1024**3)
    if estimated_gb > 30:
        print(f"Skipping: estimated {estimated_gb:.1f}GB sequence data exceeds memory budget")
        wandb.log({"status": "skipped_memory_limit", "estimated_gb": round(estimated_gb, 1)})
        return

    # Skip text/cosine modality when last_positions is unavailable (catch22/tsfresh)
    modality_components = config.modality.split('_')
    if last_positions is None and ("text" in modality_components or "cosine" in modality_components or "gemini" in modality_components):
        print(f"Skipping: text/cosine/gemini modality not supported with {feature_set} (no last_positions)")
        wandb.log({"status": "skipped_invalid_combination"})
        return

    # Feature / modality selection (only for 'full' and 'selectkbest' feature sets)
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
        if "gemini" in modality_components:
            gemini_dims = config.gemini_dims
            use_cols = list(range(4 + gemini_dims))
            df_gemini_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "gemini_video_embeddings_visual_audio_full.csv"), usecols=use_cols)
            df_gemini_index = df_gemini_raw.iloc[last_positions, 4:].reset_index(drop=True)
            del df_gemini_raw
            import gc
            gc.collect()

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
        if "gemini" in modality_components:
            selected_modalities["gemini_video"] = df_gemini_index

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

        # Check if text/cosine/gemini should be appended (e.g. rf + text)
        modality_components = config.modality.split('_')
        has_text_modality = (
            ("text" in modality_components or "cosine" in modality_components or "gemini" in modality_components)
            and last_positions is not None
        )

        # For early fusion, concatenate text/gemini features BEFORE normalization
        if has_text_modality and fusion_type == "early":
            if "text" in modality_components:
                df_text_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_embeddings_pca.csv"))
                df_text_index = df_text_raw.iloc[last_positions, 4:].reset_index(drop=True)
                df = pd.concat([df, df_text_index], axis=1)
            if "cosine" in modality_components:
                df_text_cos_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "clip_text_cosine_similarity.csv"))
                df_text_distance = df_text_cos_raw.iloc[last_positions, 4:].reset_index(drop=True)
                df = pd.concat([df, df_text_distance], axis=1)
            if "gemini" in modality_components:
                gemini_dims = config.gemini_dims
                use_cols = list(range(4 + gemini_dims))
                df_gemini_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "gemini_video_embeddings_visual_audio_full.csv"), usecols=use_cols)
                df_gemini_index = df_gemini_raw.iloc[last_positions, 4:].reset_index(drop=True)
                del df_gemini_raw
                import gc
                gc.collect()
                df = pd.concat([df, df_gemini_index], axis=1)

        if data == "norm":
            df = create_normalized_df(df)
        elif data == "pca":
            df = create_norm_pca_df(create_normalized_df(df))

        print(df)
        print(df.shape)

        # -------------------------------------------------------
        # Feature randomization (if enabled)
        # -------------------------------------------------------
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

        # Aggregate features (if enabled)
        if config.agg_features:
            info_cols = df.iloc[:, :4]
            feature_data = df.iloc[:, 4:]
            n_original = feature_data.shape[1]
            agg_df = pd.DataFrame({
                "agg_mean": feature_data.mean(axis=1).values,
                "agg_std": feature_data.std(axis=1).values,
                "agg_min": feature_data.min(axis=1).values,
                "agg_max": feature_data.max(axis=1).values,
            })
            df = pd.concat([info_cols.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
            print(f"agg_features enabled: replaced {n_original} features with 4 aggregate stats (mean, std, min, max)")
            wandb.log({"agg_features_n_original": n_original})

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
            if "gemini" in modality_components:
                gemini_dims = config.gemini_dims
                use_cols = list(range(4 + gemini_dims))
                df_gemini_raw = pd.read_csv(os.path.join(DATA_DIR, "embeddings", "gemini_video_embeddings_visual_audio_full.csv"), usecols=use_cols)
                df_gemini_index = df_gemini_raw.iloc[last_positions, 4:].reset_index(drop=True)
                del df_gemini_raw
                import gc
                gc.collect()
                df_gemini = pd.concat([info.reset_index(drop=True), df_gemini_index], axis=1)
                if data == "norm":
                    df_gemini = create_normalized_df(df_gemini)
                elif data == "pca":
                    df_gemini = create_norm_pca_df(create_normalized_df(df_gemini))
                dfs["gemini_video"] = df_gemini

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
        'name': 'brirl_transformer_binary',
        'parameters': {
            'feature_set': {'values': ['full', 'curated_features_v5_100fps', 'catch22', 'tsfresh', 'rf', 'selectkbest']},
            'modality': {'values': [
                'pose', 'facial', 'audio', 'text', 'cosine', 'gemini',
                'pose_facial', 'pose_audio', 'pose_text', 'pose_cosine', 'pose_gemini',
                'facial_audio', 'facial_text', 'facial_cosine', 'facial_gemini',
                'audio_text', 'audio_cosine', 'audio_gemini',
                'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
                'facial_audio_text', 'facial_audio_cosine', 'pose_facial_audio_cosine',
                'pose_facial_audio_text', 'pose_audio_cosine', 'pose_facial_audio_gemini',
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

            # Transformer-specific hyperparameters
            'num_heads': {'values': [2, 4, 8]},
            'num_encoder_layers': {'values': [1, 2, 3]},
            'd_model': {'values': [64, 128, 256]},
            'ff_dim': {'values': [128, 256, 512]},

            'dropout_rate': {'values': [0.0, 0.1, 0.3, 0.5]},
            'dense_units': {'values': [32, 64, 128]},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
            'learning_rate': {'values': [0.0001, 0.0005, 0.001]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'values': [100, 150, 200]},
            'loss': {'values': ["binary_crossentropy"]},

            'agg_features': {'values': [True, False]},
            'gemini_dims': {'values': [128, 256, 768, 3072]},

            'sequence_length': {'values': [5, 10, 15, 30, 60, 100, 150, 300]},

            'model': {'values': ['transformer']},
            'class': {'values': ['binary']}
        }
    }

    print(sweep_config)

    def train_wrapper():
        train()
        tf.keras.backend.clear_session()
        gc.collect()

    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_inter")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
