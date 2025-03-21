import wandb
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

        splits = create_data_splits(
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
        
        splits = {}
        for modality_key in modality_keys:
            df = modality_dfs[modality_key]
            splits[modality_key] = create_data_splits(df, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)
        
        first_modality = modality_keys[0]
        y_train_sequences = splits[first_modality][7] 
        y_val_sequences = splits[first_modality][9] 
        y_test_sequences = splits[first_modality][11] 
        
        y_train_sequences = to_categorical(y_train_sequences, num_classes=4)
        y_val_sequences = to_categorical(y_val_sequences, num_classes=4)
        y_test_sequences = to_categorical(y_test_sequences, num_classes=4)
        
        feature_inputs = []
        feature_outputs = []
        
        for modality_key in modality_keys:
            X_train_seq = splits[modality_key][6]
            
            feature_input = Input(shape=(sequence_length, X_train_seq.shape[2]))
            feature_inputs.append(feature_input)
            
            x = feature_input
            for _ in range(num_gru_layers):
                if use_bidirectional:
                    x = Bidirectional(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=kernel_regularizer))(x)
                else:
                    x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=kernel_regularizer)(x)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)
            feature_outputs.append(x)
        
        concatenated_features = concatenate(feature_outputs)
        
        x = GRU(gru_units, activation=activation, kernel_regularizer=kernel_regularizer)(concatenated_features)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        
        num_classes = y_train_sequences.shape[1] if len(y_train_sequences.shape) > 1 else len(np.unique(y_train_sequences))
        print("Num classes: ", num_classes)
        x = Dense(dense_units, activation=activation)(x)
        x = Dense(num_classes, activation="softmax")(x)
        
        model = Model(inputs=feature_inputs, outputs=x)
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
        
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
        
        df_probs = pd.DataFrame(y_predict_probs)
        table = wandb.Table(dataframe=df_probs)
        wandb.log({f"fold_{fold}_prediction_probabilities": y_predict_probs})
        wandb.log({f"fold_{fold}_prediction_probabilities_table": table})

        y_pred = np.argmax(y_predict_probs, axis=1)

        if len(y_test_sequences.shape) > 1 and y_test_sequences.shape[1] > 1:
            y_test_class_indices = np.argmax(y_test_sequences, axis=1)
        else:
            y_test_class_indices = y_test_sequences

        test_metrics = get_test_metrics(y_pred, y_test_class_indices, tolerance=1)
        
        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])
        
        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)
    
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
        
        splits = {}

        for modality_key in modality_keys:
            df = modality_dfs[modality_key]
            splits[modality_key] = create_data_splits(df, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=sequence_length)

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
                kernel_regularizer
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

        if len(y_train_sequences.shape) == 1 or y_train_sequences.shape[1] == 1:
            num_classes = len(np.unique(y_train_sequences))
            y_train_sequences = to_categorical(y_train_sequences, num_classes=num_classes)
            y_val_sequences = to_categorical(y_val_sequences, num_classes=num_classes)
            y_test_sequences = to_categorical(y_test_sequences, num_classes=num_classes)
        else:
            num_classes = y_train_sequences.shape[1]

        x = Dense(dense_units, activation=activation)(concatenated)
        output_layer = Dense(num_classes, activation="softmax")(x)
        
        model = Model(inputs=input_layers, outputs=output_layer)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
        
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

        df_probs = pd.DataFrame(y_predict_probs)
        wandb.log({f"fold_{fold}_prediction_probabilities": y_predict_probs})
        wandb.log({f"fold_{fold}_prediction_probabilities_table": wandb.Table(dataframe=df_probs)})

        y_pred = np.argmax(y_predict_probs, axis=1)

        if len(y_test_sequences.shape) > 1 and y_test_sequences.shape[1] > 1:
            y_test_class_indices = np.argmax(y_test_sequences, axis=1)
        else:
            y_test_class_indices = y_test_sequences

        test_metrics = get_test_metrics(y_pred, y_test_class_indices, tolerance=1)

        for key in test_metrics_list.keys():
            test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"fold_{fold}_metrics": test_metrics})
        print(f"Fold {fold} Test Metrics:", test_metrics)

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.run.summary.update(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)


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

    print("--------\nfeature set: ", feature_set, "\nmodality: ", modality, "\ndata: ", data, "\nfusion: ", fusion_type, "\n--------")

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
    
    modality_components = modality.split('_')
    selected_modalities = {}

    if "pose" in modality_components:
        if feature_set == "full":
            selected_modalities["pose_full"] = modalities["pose_full"]
        elif feature_set == "rf":
            selected_modalities["pose_rf"] = modalities["pose_rf"]
    
    if "facial" in modality_components:
        if feature_set == "full":
            selected_modalities["facial_full"] = modalities["facial_full"]
        elif feature_set == "stats":
            selected_modalities["facial_stats"] = modalities["facial_stats"]
        elif feature_set == "rf":
            selected_modalities["facial_rf"] = modalities["facial_rf"]
    
    if "audio" in modality_components:
        if feature_set == "full":
            selected_modalities["audio_full"] = modalities["audio_full"]
        elif feature_set == "stats":
            selected_modalities["audio_stats"] = modalities["audio_stats"]
        elif feature_set == "rf":
            selected_modalities["audio_rf"] = modalities["audio_rf"]
    
    if "text" in modality_components:
        if feature_set == "full":
            selected_modalities["text_full"] = modalities["text_full"]
    
    if len(selected_modalities) == 1:
        train_early_fusion(df, config)
    else:
        if fusion_type == "early":
            df = info
            for m in selected_modalities.values():
                df = pd.concat([df, m], axis=1)
            
            if data == "norm":
                df = create_normalized_df(df)
            elif data == "pca":
                df = create_pca_df(create_normalized_df(df))

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
                        dfs[modality_name] = create_pca_df(create_normalized_df(df_temp))
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

    feature_set = random.choice(["full", "stats", "rf"])
    
    if feature_set == "full":
        modality = random.choice(['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio', 'text', 'pose_facial_audio_text', 'pose_facial_text', 'pose_audio_text', 'facial_audio_text', 'pose_facial_audio', 'pose_facial', 'pose_audio', 'facial_audio', 'pose_text', 'facial_text', 'audio_text'])
    elif feature_set == "stats":
        modality = random.choice(['facial_audio', 'facial', 'audio'])
    elif feature_set == "rf":
        modality = random.choice(['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio'])
    
    
    sweep_config = {
        'method': 'random',
        'name': 'gru_multiclass_all_v4',
        'parameters': {
            'feature_set' : {'values': [feature_set]},
            'modality' : {'values' : [modality]},

            # 'modality_full': {'values': ['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio', 'text', 'pose_facial_audio_text']},
            # 'modality_stats': {'values': ['facial_audio', 'facial', 'audio']},
            # 'modality_rf': {'values': ['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio']},

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
            'epochs': {'value': 100},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss' : {'values' : ["categorical_crossentropy"]},
            'sequence_length' : {'values' : [60, 100, 150, 300]}
        }
        # feature set (full, stats, rf) -> modality selection (pose_facial_audio, pose, facial, etc.) -> (reg, norm, pca) -> fusion
    }

    print(sweep_config)

    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project="gru_multiclass_all_v4")
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == '__main__':
    main()
