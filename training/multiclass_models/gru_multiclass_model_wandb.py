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

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

    for fold in range(5):

        print("Fold ", fold)

        if data == "reg" or data == "norm":
            splits = create_data_splits(
                    df, "multiclass",
                    fold_no=fold,
                    num_folds=5,
                    seed_value=42,
                    sequence_length=sequence_length)
            if splits is None:
                return

        if data == "pca":
            splits = create_data_splits_pca(
                    df, "multiclass",
                    fold_no=fold,
                    num_folds=5,
                    seed_value=42,
                    sequence_length=sequence_length)
            if splits is None:
                return

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
        
        num_classes = len(np.unique(y_train))
        print("Num classes: ", num_classes)
        print("Unique labels in y_train:", np.unique(y_train))
        print("Unique labels in y_val:", np.unique(y_val))
        print("Unique labels in y_test:", np.unique(y_test))

        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(num_classes, activation="softmax"))

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

        table = wandb.Table(dataframe=df_probs)

        wandb.log({"fold_{}_prediction_probabilities".format(fold): y_predict_probs})
        wandb.log({"fold_{}_prediction_probabilities_table".format(fold): table})
        
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


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

    modality_full = config.modality_full
    modality_stats = config.modality_stats
    modality_rf = config.modality_rf

    if feature_set == "full":
        modality = modality_full
    elif feature_set == "stats":
        modality = modality_stats
    elif feature_set == "rf":
        modality = modality_rf

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

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

    for fold in range(5):

        print("Fold ", fold)

        splits_pose, splits_facial, splits_audio = None, None, None

        if feature_set == "full" or feature_set == "rf":
            if data in ["reg", "norm"]:
                if "pose" in modality:
                    splits_pose = create_data_splits(df_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                if "facial" in modality:
                    splits_facial = create_data_splits(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                if "audio" in modality:
                    splits_audio = create_data_splits(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio
            elif data == "pca":
                if "pose" in modality:
                    splits_pose = create_data_splits_pca(df_pose, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_pose, X_val_pose, X_test_pose, y_train, y_val, y_test, X_train_pose_seq, y_train_sequences, X_val_pose_seq, y_val_sequences, X_test_pose_seq, y_test_sequences, sequence_length = splits_pose
                if "facial" in modality:
                    splits_facial = create_data_splits_pca(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                if "audio" in modality:
                    splits_audio = create_data_splits_pca(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio

        elif feature_set == "stats":
            if data in ["reg", "norm"]:
                if "facial" in modality:
                    splits_facial = create_data_splits(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                if "audio" in modality:
                    splits_audio = create_data_splits(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio
            elif data == "pca":
                if "facial" in modality:
                    splits_facial = create_data_splits_pca(df_facial, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_facial, X_val_facial, X_test_facial, y_train, y_val, y_test, X_train_facial_seq, y_train_sequences, X_val_facial_seq, y_val_sequences, X_test_facial_seq, y_test_sequences, sequence_length = splits_facial
                if "audio" in modality:
                    splits_audio = create_data_splits_pca(df_audio, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
                    X_train_audio, X_val_audio, X_test_audio, y_train, y_val, y_test, X_train_audio_seq, y_train_sequences, X_val_audio_seq, y_val_sequences, X_test_audio_seq, y_test_sequences, sequence_length = splits_audio

        y_train_sequences = to_categorical(y_train_sequences, num_classes=4)
        y_val_sequences = to_categorical(y_val_sequences, num_classes=4)
        y_test_sequences = to_categorical(y_test_sequences, num_classes=4)

        feature_inputs = []

        if feature_set == "full" or feature_set == "rf":
            if "pose" in modality:
                feature_inputs.append(Input(shape=(sequence_length, X_train_pose_seq.shape[2])))
            if "facial" in modality:
                feature_inputs.append(Input(shape=(sequence_length, X_train_facial_seq.shape[2])))
            if "audio" in modality:
                feature_inputs.append(Input(shape=(sequence_length, X_train_audio_seq.shape[2])))
        
        elif feature_set == "stats":
            if "facial" in modality:
                feature_inputs.append(Input(shape=(sequence_length, X_train_facial_seq.shape[2])))
            if "audio" in modality:
                feature_inputs.append(Input(shape=(sequence_length, X_train_audio_seq.shape[2])))
        
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

        model.summary()
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

        model_checkpoint = ModelCheckpoint("../best_model.keras", monitor="val_accuracy", save_best_only=True)

        if feature_set == "full" or feature_set == "rf":
            if modality == "pose_facial":
                model_history = model.fit(
                [X_train_pose_seq, X_train_facial_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_pose_seq, X_val_facial_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2)
            elif modality == "facial_audio":
                model_history = model.fit(
                [X_train_facial_seq, X_train_audio_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_facial_seq, X_val_audio_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2)
            elif modality == "pose_facial":
                model_history = model.fit(
                [X_train_pose_seq, X_train_facial_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_pose_seq, X_val_facial_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2)
            else:
                model_history = model.fit(
                [X_train_pose_seq, X_train_facial_seq, X_train_audio_seq], y_train_sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val_pose_seq, X_val_facial_seq, X_val_audio_seq], y_val_sequences),
                # callbacks=[model_checkpoint],
                verbose=2)
        
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

        if feature_set == "full" or feature_set == "rf":
            if modality == "pose_facial":
                y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq])
            elif modality == "facial_audio":
                y_predict_probs = model.predict([X_test_facial_seq, X_test_audio_seq])
            elif modality == "pose_audio":
                y_predict_probs = model.predict([X_test_pose_seq, X_test_audio_seq])
            else:
                y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
        elif feature_set == "stats":
            y_predict_probs = model.predict([X_test_facial_seq, X_test_audio_seq])

        df_probs = pd.DataFrame(y_predict_probs)

        table = wandb.Table(dataframe=df_probs)

        wandb.log({"fold_{}_prediction_probabilities".format(fold): y_predict_probs})
        wandb.log({"fold_{}_prediction_probabilities_table".format(fold): table})
        
        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(y_test_sequences, axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def train_late_fusion(modalities, config):

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
    modality_full = config.modality_full
    modality_stats = config.modality_stats
    modality_rf = config.modality_rf

    if feature_set == "full":
        modality = modality_full
    elif feature_set == "stats":
        modality = modality_stats
    elif feature_set == "rf":
        modality = modality_rf

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

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

    for fold in range(5):

        print("Fold ", fold)

        selected_modalities = [m for m in modality if m in modalities]

        splits = {}
        input_layers = []
        outputs = []

        for modality_name in selected_modalities:
            df = modalities[modality_name]
            
            splits[modality_name] = create_data_splits(df, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

        for modality_name in selected_modalities:
            X_train_seq, X_val_seq, X_test_seq = splits[modality_name][6], splits[modality_name][8], splits[modality_name][10]
            
            input_layer = Input(shape=(sequence_length, X_train_seq.shape[2]))
            model = build_early_late_model(sequence_length, X_train_seq.shape[2], num_gru_layers, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
            
            input_layers.append(input_layer)
            outputs.append(model(input_layer))

        if len(outputs) > 1:
            concatenated = concatenate(outputs)
        else:
            concatenated = outputs[0]

        y_train_sequences = splits[selected_modalities[0]][7]
        num_classes = len(np.unique(y_train_sequences))

        x = Dense(dense_units, activation=activation)(concatenated)
        output_layer = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=input_layers, outputs=output_layer)
        model.compile(optimizer=optimizer, loss=loss, metrics=["val_accuracy"])

        train_inputs = [splits[m][6] for m in selected_modalities]
        val_inputs = [splits[m][8] for m in selected_modalities]
        test_inputs = [splits[m][10] for m in selected_modalities]

        model_history = model.fit(
            train_inputs, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, splits[selected_modalities[0]][9]),  # y_val_sequences
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

        df_probs = pd.DataFrame(y_predict_probs)
        wandb.log({f"fold_{fold}_prediction_probabilities": y_predict_probs})
        wandb.log({f"fold_{fold}_prediction_probabilities_table": wandb.Table(dataframe=df_probs)})

        y_pred = np.argmax(y_predict_probs, axis=1)
        y_test_sequences = np.argmax(splits[selected_modalities[0]][11], axis=1)

        test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


    #     if feature_set == "full" or feature_set == "rf":
    #         if modality == "pose_facial":
    #             y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq])
    #         elif modality == "facial_audio":
    #             y_predict_probs = model.predict([X_test_facial_seq, X_test_audio_seq])
    #         elif modality == "pose_audio":
    #             y_predict_probs = model.predict([X_test_pose_seq, X_test_audio_seq])
    #         else:
    #             y_predict_probs = model.predict([X_test_pose_seq, X_test_facial_seq, X_test_audio_seq])
    #     elif feature_set == "stats":
    #         y_predict_probs = model.predict([X_test_facial_seq, X_test_audio_seq])
        
    #     df_probs = pd.DataFrame(y_predict_probs)

    #     table = wandb.Table(dataframe=df_probs)

    #     wandb.log({"fold_{}_prediction_probabilities".format(fold): y_predict_probs})
    #     wandb.log({"fold_{}_prediction_probabilities_table".format(fold): table})

    #     y_pred = np.argmax(y_predict_probs, axis=1)
    #     y_test_sequences = np.argmax(y_test_sequences, axis=1)

    #     test_metrics = get_test_metrics(y_pred, y_test_sequences, tolerance=1)
    #     for key in test_metrics_list.keys():
    #         test_metrics_list[key].append(test_metrics[key])

    #     wandb.log({f"fold_{fold}_metrics": test_metrics})
    #     print(f"Fold {fold} Test Metrics:", test_metrics)
    
    # avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    # wandb.run.summary.update(avg_test_metrics)
    # print("Average Test Metrics Across All Folds:", avg_test_metrics)


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
    fusion_type = config.fusion_type

    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")
    df_text = pd.read_csv("../../preprocessing/text_embeddings.csv")
    df_text_pca = pd.read_csv("../../preprocessing/text_embeddings_pca.csv")

    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1) # action units, gaze
    df_audio_index = df.iloc[:, 63:88]
    df_text_index = df_text.iloc[:, 2:]

    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]

    df_facial_index_rf = df_rf.iloc[:, 38:]
    df_pose_index_rf = df_rf.iloc[:, 4:28]
    df_audio_index_rf = df_rf.iloc[:, 28:38]

    if feature_set == "full":
        modality = modality_full
    elif feature_set == "stats":
        modality = modality_stats
    elif feature_set == "rf":
        modality = modality_rf

    print("--------\nfeature set: ", feature_set, "\nmodality: ", modality, "\ndata: ", data, "\nfusion: ", fusion_type, "\n--------")

    modality_mapping = {
        "pose": pd.concat([info, df_pose_index], axis=1),
        "facial": pd.concat([info, df_facial_index], axis=1),
        "audio": pd.concat([info, df_audio_index], axis=1),
        "pose_facial": pd.concat([info, df_pose_index, df_facial_index], axis=1),
        "pose_audio": pd.concat([info, df_pose_index, df_audio_index], axis=1),
        "facial_audio": pd.concat([info, df_facial_index, df_audio_index], axis=1),
        "pose_facial_audio": pd.concat([info, df_pose_index, df_facial_index, df_audio_index], axis=1),
    }

    modality_mapping_stats = {
        "facial": pd.concat([info, df_facial_index_stats], axis=1),
        "audio": pd.concat([info, df_audio_index_stats], axis=1),
        "pose_facial_audio": pd.concat([info, df_facial_index_stats, df_audio_index_stats], axis=1),
    }

    modality_mapping_rf = {
        "pose": pd.concat([info, df_pose_index_rf], axis=1),
        "facial": pd.concat([info, df_facial_index_rf], axis=1),
        "audio": pd.concat([info, df_audio_index_rf], axis=1),
        "pose_facial": pd.concat([info, df_pose_index_rf, df_facial_index_rf], axis=1),
        "pose_audio": pd.concat([info, df_pose_index_rf, df_audio_index_rf], axis=1),
        "facial_audio": pd.concat([info, df_facial_index_rf, df_audio_index_rf], axis=1),
        "pose_facial_audio": pd.concat([info, df_facial_index_rf, df_pose_index_rf, df_audio_index_rf], axis=1),
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
    
    def create_pca_df(df):
        participant_frames_labels = df.iloc[:, :4]

        x = df.iloc[:, 4:]
        x = StandardScaler().fit_transform(x.values)

        pca = PCA(n_components=0.90)
        principal_components = pca.fit_transform(x)
        print(principal_components.shape)

        principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
        principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

        return principal_df
    
    modalities = {

        "pose_full" : df_pose_index,
        "pose_rf" : df_pose_index_rf,

        "facial_full" : df_facial_index,
        "facial_stats" : df_facial_index_stats,
        "facial_rf" : df_facial_index_rf,

        "audio_full" : df_audio_index,
        "audio_stats" : df_audio_index_stats,
        "audio_rf" : df_audio_index_rf,

        "text_full" : df_text_index,

    }
    
    selected_modalities = {}

    def process_data(feature_df, info, data_type):
        combined_df = pd.concat([info, feature_df], axis=1)
        if data_type == "norm":
            return create_normalized_df(combined_df)
        elif data_type == "pca":
            return create_pca_df(create_normalized_df(combined_df))
        return combined_df  # Default to raw data

    selected_modalities = {}

    if fusion_type == "early":
        # Early fusion: Concatenate all modalities first, then normalize/PCA
        all_features = []

        for m in modality:
            if "pose" in m:
                all_features.append(df_pose_index)
            elif "facial" in m:
                all_features.append(df_facial_index)
            elif "audio" in m:
                all_features.append(df_audio_index)
            elif "text" in m:
                all_features.append(df_text_index)

        combined_df = pd.concat([info] + all_features, axis=1)

        if data == "norm":
            selected_modalities["early_fusion"] = create_normalized_df(combined_df)
        elif data == "pca":
            selected_modalities["early_fusion"] = create_pca_df(create_normalized_df(combined_df))
        else:
            selected_modalities["early_fusion"] = combined_df

    elif fusion_type == "intermediate":
        # Intermediate fusion: Process each modality independently, then concatenate later
        processed_modalities = []

        for m in modality:
            if "pose" in m:
                processed_modalities.append(process_data(df_pose_index, info, data))
            elif "facial" in m:
                processed_modalities.append(process_data(df_facial_index, info, data))
            elif "audio" in m:
                processed_modalities.append(process_data(df_audio_index, info, data))
            elif "text" in m:
                processed_modalities.append(pd.concat([info, df_text_index], axis=1))  # No PCA for text

        selected_modalities["intermediate_fusion"] = pd.concat(processed_modalities, axis=1)

    elif fusion_type == "late":
        # Late fusion: Each modality is processed separately
        for m in modality:
            if "pose" in m:
                selected_modalities["pose"] = process_data(df_pose_index, info, data)
            elif "facial" in m:
                selected_modalities["facial"] = process_data(df_facial_index, info, data)
            elif "audio" in m:
                selected_modalities["audio"] = process_data(df_audio_index, info, data)
            elif "text" in m:
                selected_modalities["text"] = pd.concat([info, df_text_index], axis=1)  # No PCA for text


    
    for m in modality:
        if "pose" in m:
            if feature_set == "full":
                if data == "norm":
                    selected_modalities["pose_full"] = create_normalized_df(pd.concat([info, df_pose_index], axis=1))
                elif data == "pca":
                    selected_modalities["pose_full"] = create_pca_df(create_normalized_df(pd.concat([info, df_pose_index], axis=1)))
                elif data == "reg":
                    selected_modalities["pose_full"] = pd.concat([info, df_pose_index], axis=1)
            elif feature_set == "rf":
                selected_modalities["pose_rf"] = pd.concat([info, df_pose_index_rf], axis=1)
 
        elif "facial" in m:
            if feature_set == "full":
                selected_modalities["facial_full"] = pd.concat([info, df_facial_index], axis=1)
            elif feature_set == "stats":
                selected_modalities["facial_stats"] = pd.concat([info, df_facial_index_stats], axis=1)
            elif feature_set == "rf":
                selected_modalities["facial_rf"] = pd.concat([info, df_facial_index_rf], axis=1)

        elif "audio" in m:
            if feature_set == "full":
                selected_modalities["audio_full"] = pd.concat([info, df_audio_index], axis=1)
            elif feature_set == "stats":
                selected_modalities["audio_stats"] = pd.concat([info, df_audio_index_stats], axis=1)
            elif feature_set == "rf":
                selected_modalities["audio_rf"] = pd.concat([info, df_audio_index_rf], axis=1)

        elif "text" in m:
            selected_modalities["text_full"] = pd.concat([info, df_text_index], axis=1)


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

    if feature_set == "full": #'pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio'

        if fusion_type == "early":
            train_early_fusion(get_modality_data(modality_full, data), config)

        elif fusion_type == "intermediate":

            if modality == "pose_facial":
                train_intermediate_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    pd.DataFrame(),
                    config
                )

            elif modality == "pose_audio":
                train_intermediate_fusion(
                    get_modality_data("pose", data),
                    pd.DataFrame(),
                    get_modality_data("audio", data),
                    config
                )

            elif modality == "facial_audio":
                train_intermediate_fusion(
                    pd.DataFrame(),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config
                )
            
            elif modality == "pose_facial_audio":
                train_intermediate_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config,
                )
            
            else:
                train_single_modality_model(get_modality_data(modality, data), config)

        elif fusion_type == "late":

            if modality == "pose_facial":
                train_late_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    pd.DataFrame(),
                    config
                )

            elif modality == "pose_audio":
                train_late_fusion(
                    get_modality_data("pose", data),
                    pd.DataFrame(),
                    get_modality_data("audio", data),
                    config
                )

            elif modality == "facial_audio":
                train_late_fusion(
                    pd.DataFrame(),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config
                )
            
            elif modality == "pose_facial_audio":
                train_late_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config,
                )
            
            else:
                train_single_modality_model(get_modality_data(modality, data), config)

        
    elif feature_set == "stats": #'modality_stats' : {'values': ['pose_facial_audio', 'facial', 'audio']},

        if fusion_type == "early":
            train_early_fusion(get_modality_data(modality, data), config)
        
        elif fusion_type == "intermediate":
            if modality == "facial_audio":
                train_intermediate_fusion(
                    pd.DataFrame(),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config,
                )
            else:
                train_single_modality_model(get_modality_data(modality, data), config)

        elif fusion_type == "late" and modality == "facial_audio":
            if modality == "facial_audio":
                train_late_fusion(
                    pd.DataFrame(),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config,
                )
            else:
                train_single_modality_model(get_modality_data(modality, data), config)
    
    elif feature_set == "rf":

        if fusion_type == "early":
            train_early_fusion(get_modality_data(modality, data), config)

        elif fusion_type == "intermediate":
            if modality == "pose_facial":
                train_intermediate_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    pd.DataFrame(),
                    config
                )

            elif modality == "pose_audio":
                train_intermediate_fusion(
                    get_modality_data("pose", data),
                    pd.DataFrame(),
                    get_modality_data("audio", data),
                    config
                )

            elif modality == "facial_audio":
                train_intermediate_fusion(
                    pd.DataFrame(),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config
                )
            
            elif modality == "pose_facial_audio":
                train_intermediate_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config
                )
            
            else:
                train_single_modality_model(get_modality_data(modality, data), config)

        elif fusion_type == "late":

            if modality == "pose_facial":
                train_late_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    pd.DataFrame(),
                    config
                )

            elif modality == "pose_audio":
                train_late_fusion(
                    get_modality_data("pose", data),
                    pd.DataFrame(),
                    get_modality_data("audio", data),
                    config
                )

            elif modality == "facial_audio":
                train_late_fusion(
                    pd.DataFrame(),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config
                )
            
            elif modality == "pose_facial_audio":
                train_late_fusion(
                    get_modality_data("pose", data),
                    get_modality_data("facial", data),
                    get_modality_data("audio", data),
                    config,
                )
            
            else:
                train_single_modality_model(get_modality_data(modality, data), config)

def main():
    
    sweep_config = {
        'method': 'random',
        'name': 'gru_multiclass_all_v2',
        'parameters': {
            'feature_set' : {'values': ["full", "stats", "rf"]},
            'modality_full' : {'values': ['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio']},
            'modality_stats' : {'values': ['facial_audio', 'facial', 'audio']},
            'modality_rf' : {'values': ['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio']},
            'data' : {'values' : ["reg", "norm", "pca"]},
            'fusion_type': {'values': [ 'intermediate', 'late']},
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
        # feature set (full, stats, rf) -> modality selection (pose_facial_audio, pose, facial, etc.) -> (reg, norm, pca) -> fusion
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="gru_multiclass_all_v2")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
