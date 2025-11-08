import wandb
import numpy as np
import pandas as pd
import random
import gc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2, l1, l2
from keras.metrics import Precision, Recall, AUC

import tensorflow as tf
from create_data_splits import create_data_splits_intra_balanced as create_data_splits
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
    train_ratio = config.train_ratio
    split_strategy = config.split_strategy

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
    
    unique_sessions = df['participant'].unique()
    print("Total folds:", len(unique_sessions))

    for fold_no in range(len(unique_sessions)):
        print(f"\n=== Fold {fold_no} / session {unique_sessions[fold_no]} ===")

        splits = create_data_splits(
            df, label_column='multiclass_label', split_strategy=split_strategy,
            fold_no=fold_no,
            train_ratio=train_ratio,
            test_ratio=0.2,
            seed_value=42,
            sequence_length=sequence_length
        )
        if splits is None:
            print(f"[Fold {fold_no}] Invalid split. Skipping…")
            continue
        
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         X_train_sequences, y_train_sequences,
         X_val_sequences, y_val_sequences,
         X_test_sequences, y_test_sequences,
         sequence_length) = splits

        # Remap class labels to contiguous indices
        unique_classes = np.unique(y_train_sequences)
        class_mapping = {old: new for new, old in enumerate(unique_classes)}

        # Build a lookup array where index = original label, value = remapped label
        max_label = max(unique_classes)
        mapping_array = np.zeros(max_label + 1, dtype=int)
        for old, new in class_mapping.items():
            mapping_array[old] = new

        # Apply mapping
        y_train_sequences = mapping_array[y_train_sequences]
        y_val_sequences   = mapping_array[y_val_sequences]
        y_test_sequences  = mapping_array[y_test_sequences]

        print("After mapping:")
        print("Train:", np.unique(y_train_sequences))
        print("Val:",   np.unique(y_val_sequences))
        print("Test:",  np.unique(y_test_sequences))

        if kernel_regularizer == "l1":
            reg = l1(0.01)
        elif kernel_regularizer == "l2":
            reg = l2(0.01)
        elif kernel_regularizer == "l1_l2":
            reg = l1_l2(l1=0.01,l2=0.01)
        else:
            reg = None

        input_shape = X_train_sequences.shape[2]

        model = build_early_late_model(
            sequence_length, input_shape,
            num_gru_layers, gru_units,
            activation, use_bidirectional,
            dropout, reg
        )

        # num_classes = 4
        num_classes = len(np.unique(y_train_sequences))
        print("Num classes: ", num_classes)
        print("Unique labels in y_train_sequences:", np.unique(y_train_sequences))
        print("Unique labels in y_val:", np.unique(y_val))
        print("Unique labels in y_test:", np.unique(y_test))

        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(num_classes, activation="softmax"))

        model.summary()

        if optimizer == 'adam':
            optim = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optim = optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optim = optimizers.RMSprop(learning_rate=learning_rate)

        # multiclass_metrics = [
        #     'accuracy',
        #     Precision(name='precision'),
        #     Recall(name='recall'),
        #     AUC(name='auc')
        # ]

        model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
        
        model_history = model.fit(
            X_train_sequences, y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_sequences, y_val_sequences),
            verbose=2
        )

        for epoch in range(len(model_history.history['loss'])):
            metrics = {
                'fold': fold_no,
                'epoch': epoch,
                'loss': model_history.history['loss'][epoch],
                'val_loss': model_history.history['val_loss'][epoch]
            }
            for k in ['accuracy', 'val_accuracy',
                      'precision', 'val_precision',
                      'recall', 'val_recall',
                      'auc', 'val_auc']:
                if k in model_history.history:
                    metrics[k] = model_history.history[k][epoch]
            wandb.log(metrics)

        y_predict_probs = model.predict(X_test_sequences)
        y_predict_probs_clean = np.nan_to_num(y_predict_probs, nan=0.0)

        print("Pred shape:", y_predict_probs_clean.shape)
        print("y_val_sequences shape:", y_val_sequences.shape)
        print("y_test_sequences shape:", y_test_sequences.shape)

        df_probs = pd.DataFrame(y_predict_probs_clean)

        table = wandb.Table(dataframe=df_probs)

        wandb.log({"participant_{}_prediction_probabilities".format(fold_no): y_predict_probs_clean})
        wandb.log({"participant_{}_prediction_probabilities_table".format(fold_no): table})
        
        y_pred = np.argmax(y_predict_probs_clean, axis=1)

        y_test_class_indices = y_test_sequences
        
        cm = confusion_matrix(y_test_class_indices, y_pred)

        unique, counts = np.unique(y_test, return_counts=True)
        print("\nTest label distribution:")
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count}")

        print("\nConfusion Matrix (Multiclass):")
        #print(pd.DataFrame(cm, index=["Actual 0", "Actual 1", "Actual 2", "Actual 3"], columns=["Pred 0", "Pred 1", "Pred2 2", "Pred 3"]))

        print(pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in range(num_classes)],
            columns=[f"Pred {c}" for c in range(num_classes)]
        ))


        wandb.log({f"fold_{fold_no}_confusion_matrix": cm})

        test_metrics = get_test_metrics(y_pred, y_test_class_indices, tolerance=1)

        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"participant_{fold_no}_metrics": test_metrics})
        print(f"Fold {fold_no} Test Metrics:", test_metrics)

        del model, model_history, X_train_sequences, X_val_sequences, X_test_sequences
        gc.collect()
        tf.keras.backend.clear_session()
    
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
    train_ratio = config.train_ratio
    split_strategy = config.split_strategy

    modality_keys = list(modality_dfs.keys())

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

    unique_sessions = modality_dfs[next(iter(modality_dfs))]['participant'].unique()
    
    print("Total folds:", len(unique_sessions))

    if kernel_regularizer == "l1":
        reg = l1(0.01)
    elif kernel_regularizer == "l2":
        reg = l2(0.01)
    elif kernel_regularizer == "l1_l2":
        reg = l1_l2(l1=0.01,l2=0.01)
    else:
        reg = None

    for fold_no, participant in enumerate(unique_sessions):
        print(f"\n=== Fold {fold_no} / session {participant} ===")

        splits_valid = True
        splits = {}

        for modality_key in modality_keys:
            df = modality_dfs[modality_key]

            splits[modality_key] = create_data_splits(
                df, label_column='multiclass_label', split_strategy=split_strategy,
                fold_no=fold_no,
                train_ratio=train_ratio,
                test_ratio=0.20,
                seed_value=42,
                sequence_length=sequence_length
            )

            if splits[modality_key] is None:
                print(f"[Fold {fold_no}] Invalid split for {modality_key}. Skipping...")
                splits_valid = False
                break
            if splits[modality_key][6].shape[0] == 0 or splits[modality_key][7].shape[0] == 0:
                print(f"[Fold {fold_no}] Empty split for {modality_key}. Skipping...")
                splits_valid = False
                break

        if not splits_valid:
            continue
        
        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7] 
        y_val_sequences = splits[first_modality][9] 
        y_test_sequences = splits[first_modality][11] 

        # Remap class labels to contiguous indices
        unique_classes = np.unique(y_train_sequences)
        class_mapping = {old: new for new, old in enumerate(unique_classes)}

        # Build a lookup array where index = original label, value = remapped label
        max_label = max(unique_classes)
        mapping_array = np.zeros(max_label + 1, dtype=int)
        for old, new in class_mapping.items():
            mapping_array[old] = new

        # Apply mapping
        y_train_sequences = mapping_array[y_train_sequences]
        y_val_sequences   = mapping_array[y_val_sequences]
        y_test_sequences  = mapping_array[y_test_sequences]

        print("After mapping:")
        print("Train:", np.unique(y_train_sequences))
        print("Val:",   np.unique(y_val_sequences))
        print("Test:",  np.unique(y_test_sequences))
        
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

        #num_classes = 4
        num_classes = len(np.unique(y_train_sequences))
        print("Num classes: ", num_classes)
        x = Dense(dense_units, activation=activation)(x)
        x = Dense(num_classes, activation="softmax")(x)

        if optimizer == 'adam':
            optim = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optim = optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optim = optimizers.RMSprop(learning_rate=learning_rate)
        
        model = Model(inputs=feature_inputs, outputs=x)
        model.summary()

        # multiclass_metrics = [
        #     'accuracy',
        #     Precision(name='precision'),
        #     Recall(name='recall'),
        #     AUC(name='auc')
        # ]

        model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
        
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
                'participant': participant,
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

        print("Pred shape:", y_predict_probs_clean.shape)
        print("y_val_sequences shape:", y_val_sequences.shape)
        print("y_test_sequences shape:", y_test_sequences.shape)
        
        df_probs = pd.DataFrame(y_predict_probs_clean)
        table = wandb.Table(dataframe=df_probs)
        wandb.log({f"participant_{participant}_prediction_probabilities": y_predict_probs_clean})
        wandb.log({f"participant_{participant}_prediction_probabilities_table": table})
        
        y_pred = np.argmax(y_predict_probs_clean, axis=1)

        y_test_class_indices = y_test_sequences

        cm = confusion_matrix(y_test_class_indices, y_pred)

        unique, counts = np.unique(y_test_class_indices, return_counts=True)
        print("\nTest label distribution:")
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count}")

        print("\nConfusion Matrix (Multiclass):")
        #print(pd.DataFrame(cm, index=["Actual 0", "Actual 1", "Actual 2", "Actual 3"], columns=["Pred 0", "Pred 1", "Pred2 2", "Pred 3"]))
        print(pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in range(num_classes)],
            columns=[f"Pred {c}" for c in range(num_classes)]
        ))


        wandb.log({f"fold_{fold_no}_confusion_matrix": cm})

        wandb.log({f"participant_{fold_no}_confusion_matrix": cm.tolist()})

        test_metrics = get_test_metrics(y_pred, y_test_class_indices, tolerance=1)
        
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])
        
        wandb.log({f"participant_{participant}_metrics": test_metrics})
        print(f"Fold {participant} Test Metrics:", test_metrics)

        del model, model_history
        for m in modality_keys:
            del splits[m]
        gc.collect()
        tf.keras.backend.clear_session()
    
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
    train_ratio = config.train_ratio
    split_strategy = config.split_strategy

    modality_keys = list(modality_dfs.keys())

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

    unique_sessions = modality_dfs[next(iter(modality_dfs))]['participant'].unique()
    
    print("Total folds:", len(unique_sessions))

    if kernel_regularizer == "l1":
        reg = l1(0.01)
    elif kernel_regularizer == "l2":
        reg = l2(0.01)
    elif kernel_regularizer == "l1_l2":
        reg = l1_l2(l1=0.01,l2=0.01)
    else:
        reg = None
        
    for fold_no, participant in enumerate(unique_sessions):
        print(f"\n=== Fold {fold_no} / session {participant} ===")

        splits_valid = True
        splits = {}

        for modality_key in modality_keys:
            df = modality_dfs[modality_key]

            splits[modality_key] = create_data_splits(
                df, label_column='multiclass_label', split_strategy=split_strategy,
                fold_no=fold_no,
                train_ratio=train_ratio,
                test_ratio=0.20,
                seed_value=42,
                sequence_length=sequence_length
            )

            if splits[modality_key] is None:
                print(f"[Fold {fold_no}] Invalid split for {modality_key}. Skipping...")
                splits_valid = False
                break
            if splits[modality_key][6].shape[0] == 0 or splits[modality_key][7].shape[0] == 0:
                print(f"[Fold {fold_no}] Empty split for {modality_key}. Skipping...")
                splits_valid = False
                break

        if not splits_valid:
            continue
        
        input_layers = []
        outputs = []

        # Build each modality stream directly in functional API
        for modality_key in modality_keys:
            X_train_seq = splits[modality_key][6]
            
            input_layer = Input(shape=(sequence_length, X_train_seq.shape[2]))
            input_layers.append(input_layer)
            
            # Build the GRU layers directly instead of using Sequential
            x = input_layer
            
            if num_gru_layers == 1:
                if use_bidirectional:
                    x = Bidirectional(GRU(gru_units, activation=activation, kernel_regularizer=reg))(x)
                else:
                    x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(x)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)
            else:
                for i in range(num_gru_layers - 1):
                    if use_bidirectional:
                        x = Bidirectional(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg))(x)
                    else:
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                    x = Dropout(dropout)(x)
                    x = BatchNormalization()(x)
                
                # Final GRU layer without return_sequences
                if use_bidirectional:
                    x = Bidirectional(GRU(gru_units, activation=activation, kernel_regularizer=reg))(x)
                else:
                    x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(x)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)
            
            outputs.append(x)
        
        if len(outputs) > 1:
            concatenated = concatenate(outputs)
        else:
            concatenated = outputs[0]
        
        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7]
        y_val_sequences = splits[first_modality][9]
        y_test_sequences = splits[first_modality][11]

        # Remap class labels to contiguous indices
        unique_classes = np.unique(y_train_sequences)
        class_mapping = {old: new for new, old in enumerate(unique_classes)}

        # Build a lookup array where index = original label, value = remapped label
        max_label = max(unique_classes)
        mapping_array = np.zeros(max_label + 1, dtype=int)
        for old, new in class_mapping.items():
            mapping_array[old] = new

        # Apply mapping
        y_train_sequences = mapping_array[y_train_sequences]
        y_val_sequences   = mapping_array[y_val_sequences]
        y_test_sequences  = mapping_array[y_test_sequences]

        print("After mapping:")
        print("Train:", np.unique(y_train_sequences))
        print("Val:",   np.unique(y_val_sequences))
        print("Test:",  np.unique(y_test_sequences))

        num_classes = len(np.unique(y_train_sequences))

        x = Dense(dense_units, activation=activation)(concatenated)
        output_layer = Dense(num_classes, activation="softmax")(x)
        
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
        
        # Properly configure metrics for multiclass classification
        # multiclass_metrics = [
        #     'accuracy',
        #     Precision(name='precision'),
        #     Recall(name='recall'),
        #     AUC(name='auc')
        # ]
        
        model.compile(optimizer=optim, loss=loss, metrics=["accuracy"])
        
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
                'participant': participant,
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

        print("Pred shape:", y_predict_probs_clean.shape)
        print("y_val_sequences shape:", y_val_sequences.shape)
        print("y_test_sequences shape:", y_test_sequences.shape)

        df_probs = pd.DataFrame(y_predict_probs_clean)
        wandb.log({f"participant_{participant}_prediction_probabilities": y_predict_probs_clean})
        wandb.log({f"participant_{participant}_prediction_probabilities_table": wandb.Table(dataframe=df_probs)})

        y_pred = np.argmax(y_predict_probs_clean, axis=1)

        y_test_class_indices = y_test_sequences

        cm = confusion_matrix(y_test_class_indices, y_pred)

        unique, counts = np.unique(y_test_class_indices, return_counts=True)
        print("\nTest label distribution:")
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count}")

        print("\nConfusion Matrix (Multiclass):")
        print(pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in range(num_classes)],
            columns=[f"Pred {c}" for c in range(num_classes)]
        ))

        wandb.log({f"fold_{fold_no}_confusion_matrix": cm})
        wandb.log({f"participant_{participant}_confusion_matrix": cm.tolist()})

        test_metrics = get_test_metrics(y_pred, y_test_class_indices, tolerance=1)

        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"participant_{participant}_metrics": test_metrics})
        print(f"Fold {participant} Test Metrics:", test_metrics)

        del model, model_history
        gc.collect()
        tf.keras.backend.clear_session()

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


def validate_modality_feature_combination(modality, feature_set):
    modality_components = modality.split('_')
    
    # Text only works with 'full' feature set
    if 'text' in modality_components and feature_set != 'full':
        return False
        
    # 'stats' feature set only works with 'facial', 'audio', and their combination
    if feature_set == 'stats':
        valid_components = ['facial', 'audio']
        for component in modality_components:
            if component not in valid_components:
                return False
    
    # 'rf' feature set doesn't work with 'text'
    if feature_set == 'rf' and 'text' in modality_components:
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

    # Validate modality and feature_set combination
    is_valid_combination = validate_modality_feature_combination(config.modality, config.feature_set)
    
    if not is_valid_combination:
        print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
        # Log that this was skipped
        wandb.log({"status": "skipped_invalid_combination"})
        return

    data = config.dataset
    fusion_type = config.fusion_type

    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")
    df_text = pd.read_csv("../../preprocessing/clip_text_embeddings.csv")
    df_text_pca = pd.read_csv("../../preprocessing/clip_text_embeddings_pca.csv")

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

    modalities = {
        "pose_full": df_pose_index,
        "pose_rf": df_pose_index_rf,
        "facial_full": df_facial_index,
        "facial_stats": df_facial_index_stats,
        "facial_rf": df_facial_index_rf,
        "audio_full": df_audio_index,
        "audio_stats": df_audio_index_stats,
        "audio_rf": df_audio_index_rf,
        "text_full": df_text_index,
    }

    modality_components = config.modality.split('_')
    selected_modalities = {}

    feature_set = config.feature_set

    if "pose" in modality_components:
        selected_modalities["pose_" + feature_set] = modalities["pose_" + feature_set]
    
    if "facial" in modality_components:
        selected_modalities["facial_" + feature_set] = modalities["facial_" + feature_set]

    if "audio" in modality_components:
        selected_modalities["audio_" + feature_set] = modalities["audio_" + feature_set]

    if "text" in modality_components:
        selected_modalities["text_full"] = modalities["text_full"]

    df = info
    
    # for m in selected_modalities.values():
    #     df = pd.concat([df, m], axis=1)

    # if config.dataset == "norm":
    #     df = create_normalized_df(df)
    # elif config.dataset == "pca":
    #     df = create_norm_pca_df(df)
    
    if fusion_type == "early":
        df = info
        for m in selected_modalities.values():
            df = pd.concat([df, m], axis=1)
        
        if data == "norm":
            df = create_normalized_df(df)
        elif data == "pca":
            df = create_norm_pca_df(create_normalized_df(df))

        print(df)
        print(df.shape)
        
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
                    dfs[modality_name] = df_text_pca
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


def main():

    sweep_config = {
        'method': 'random',
        'name': 'multiclass_intra_balanced',
        'parameters': {
            'feature_set': {'values': ['full', 'stats', 'rf']},
            'modality': {'values': [
                'pose', 'facial', 'audio', 'text',
                'pose_facial', 'pose_audio', 'pose_text',
                'facial_audio', 'facial_text',
                'audio_text',
                'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
                'facial_audio_text',
                'pose_facial_audio_text',
            ]},

            'dataset' : {'values' : ["reg", "norm", "pca"]},
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
            'epochs': {'value': 50},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss' : {'values' : ["sparse_categorical_crossentropy"]},
            
            'sequence_length' : {'values' : [5, 10, 15, 30, 60]},

            'train_ratio' : {'values' : [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},

            'split_strategy' : {'values' : ["multiclass", "multiclass_exclude_neutral"]},

            'model' : {'values': ['gru']}

        }
        # feature set (full, stats, rf) -> modality selection (pose_facial_audio, pose, facial, etc.) -> (reg, norm, pca) -> fusion
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="multiclass_intra_balanced")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
