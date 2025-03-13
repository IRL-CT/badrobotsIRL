import wandb
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1_l2, l1, l2
from keras.utils import to_categorical
import tensorflow as tf
from create_data_splits import create_data_splits, create_data_splits_pca
from get_metrics import get_test_metrics

def train_single_modality_model(df, config):

    print("training lstm single modality model")

    num_lstm_layers = config.num_lstm_layers
    lstm_units = config.lstm_units
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

        splits = None

        if data == "reg" or data == "norm":
            splits = create_data_splits(
                df, "binary",
                fold_no=fold,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
            if splits is None:
                return

        elif data == "pca":
            splits = create_data_splits_pca(
                df, "binary",
                fold_no=fold,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length)
            if splits is None:
                return

        if splits is None:
            raise ValueError(f"Failed to create data splits for data type '{data}'.")

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

        model = Sequential()
        model.add(Input(shape=(sequence_length, input_shape)))

        if num_lstm_layers == 1:
            if use_bidirectional:
                model.add(Bidirectional(LSTM(lstm_units, activation=activation, kernel_regularizer=reg)))
            else:
                model.add(LSTM(lstm_units, activation=activation, kernel_regularizer=reg))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())
        else:
            for _ in range(num_lstm_layers - 1):
                if use_bidirectional:
                    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True, activation=activation, kernel_regularizer=reg)))
                else:
                    model.add(LSTM(lstm_units, return_sequences=True, activation=activation, kernel_regularizer=reg))
                model.add(Dropout(dropout))
                model.add(BatchNormalization())

            if use_bidirectional:
                model.add(Bidirectional(LSTM(lstm_units, activation=activation)))
            else:
                model.add(LSTM(lstm_units, activation=activation))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())
        
        if loss == "categorical_crossentropy":
            num_classes = len(np.unique(y_train_sequences))
            model.add(Dense(dense_units, activation=activation))
            model.add(Dense(num_classes, activation="softmax"))
        else:
            model.add(Dense(dense_units, activation=activation))
            model.add(Dense(1, activation="sigmoid"))

        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

        model.summary()
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

        #model_checkpoint = ModelCheckpoint("../best_model.keras", monitor="val_accuracy", save_best_only=True)

        # early_stopping = EarlyStopping(
        #     monitor="val_accuracy",
        #     patience=10,
        #     min_delta=0.001,
        #     restore_best_weights=True,
        #     verbose=1
        # )
        
        model_history = model.fit(
            X_train_sequences, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_sequences, y_val_sequences),
            #callbacks=[early_stopping],
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

        df_probs = pd.DataFrame(y_predict_probs)

        print(df_probs)

        df_probs = pd.DataFrame(y_predict_probs, columns=[f"class_{i}" for i in range(y_predict_probs.shape[1])])

        print(df_probs)

        table = wandb.Table(dataframe=df_probs)

        wandb.log({"fold_{}_prediction_probabilities".format(fold): y_predict_probs})
        wandb.log({"fold_{}_prediction_probabilities_table".format(fold): table})
        
        if loss == "categorical_crossentropy":
            y_pred = np.argmax(y_predict_probs, axis=1)
            y_test_sequences = np.argmax(y_test_sequences, axis=1)
        else:
            y_pred = (y_predict_probs > 0.5).astype(int).flatten()
            y_test_sequences = y_test_sequences.astype(int).flatten()

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)

def train():

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    modality = config.modality
    feature_set = config.feature_set
    data = config.data

    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")

    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
    df_audio_index = df.iloc[:, 63:88]

    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]

    df_facial_index_rf = df_rf.iloc[:, 38:]
    df_pose_index_rf = df_rf.iloc[:, 4:28]
    df_audio_index_rf = df_rf.iloc[:, 28:38]

    modality_mapping = {
        "pose": pd.concat([info, df_pose_index], axis=1),
        "facial": pd.concat([info, df_facial_index], axis=1),
        "audio": pd.concat([info, df_audio_index], axis=1)
    }

    modality_mapping_stats = {
        "facial": pd.concat([info, df_facial_index_stats], axis=1),
        "audio": pd.concat([info, df_audio_index_stats], axis=1)
    }

    modality_mapping_rf = {
        "pose": pd.concat([info, df_pose_index_rf], axis=1),
        "facial": pd.concat([info, df_facial_index_rf], axis=1),
        "audio": pd.concat([info, df_audio_index_rf], axis=1)
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

        elif feature_set == "rf":
            df = modality_mapping_rf.get(modality)

        if data != "reg":
            df = create_normalized_df(df)

        return df

    train_single_modality_model(get_modality_data(modality, data), config)

def main():

    modality = "pose"
    
    sweep_config = {
        'method': 'random',
        'name': f'lstm_binary_{modality}_v1',
        'parameters': {
            'modality' : {'value': modality},

            'feature_set' : {'values': ["full", "stats", "rf"]},
            'data' : {'values' : ["reg", "norm", "pca"]},

            'use_bidirectional': {'values': [True, False]},
            'num_lstm_layers': {'values': [1, 2, 3]},
            'lstm_units': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'dense_units': {'values': [32, 64, 128]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
            'learning_rate': {'values': [0.001, 0.01, 0.005]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'value': 100},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss' : {'values' : ["binary_crossentropy", "categorical_crossentropy"]},
            'sequence_length' : {'values' : [30, 60, 90]}
        }
        # feature set (full, stats, rf) -> modality selection (combined, pose, facial, etc.) -> (reg, norm, pca) -> fusion
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project=f"lstm_binary_{modality}_v1")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
