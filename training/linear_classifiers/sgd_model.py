import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import wandb
from itertools import product
from get_metrics import get_test_metrics
import random
from create_data_splits import create_data_splits

def train():
    wandb.init()
    config = wandb.config
    print(config)
    
    # Validate modality and feature_set combination
    is_valid_combination = validate_modality_feature_combination(config.modality, config.feature_set)
    
    if not is_valid_combination:
        print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
        # Log that this was skipped
        wandb.log({"status": "skipped_invalid_combination"})
        return
    
    seed_value = 42

    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")
    df_text = pd.read_csv("../../preprocessing/clip_text_embeddings_pca.csv")
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

    # Select dataset and modalities
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
    
    for m in selected_modalities.values():
        df = pd.concat([df, m], axis=1)

    if config.dataset == "norm":
        df = create_normalized_df(df)
    elif config.dataset == "pca":
        df = create_norm_pca_df(df)

    print(df)
    print(df.shape)
    
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
        splits = create_data_splits(
            df, config.binary_multiclass,
            fold_no=fold,
            num_folds=5,
            seed_value=42,
            sequence_length=1)
        X_train, X_val, X_test, y_train, y_val, y_test, _, _, _, _, _, _, _ = splits
            
        # Balance training dataset
        smote = SMOTE(random_state=seed_value) 
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Use SGD parameters from config
        # Initialize SGDClassifier with proper hyperparameters from sweep config
        sgd = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=config.alpha if hasattr(config, 'alpha') else 0.0001,
            max_iter=config.max_iter if hasattr(config, 'max_iter') else 1000,
            tol=config.tol if hasattr(config, 'tol') else 1e-3,
            random_state=seed_value
        )
        
        sgd.fit(X_train_balanced, y_train_balanced)
    
        y_pred = sgd.predict(X_test)

        # Logging prediction probabilities
        y_pred_proba = sgd.predict_proba(X_test)

        df_probs = pd.DataFrame(y_pred_proba, columns=[f"prob_class_{i}" for i in range(y_pred_proba.shape[1])])
        df_probs["y_pred"] = y_pred
        df_probs["y_true"] = y_test

        table = wandb.Table(dataframe=df_probs)

        wandb.log({
            f"fold_{fold}_prediction_probabilities": wandb.Histogram(y_pred_proba),
            f"fold_{fold}_prediction_probabilities_table": table
        })

        test_metrics = get_test_metrics(y_pred, y_test, tolerance=1)
        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        test_metrics = {f"t{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)
        print(test_metrics)

        print(confusion_matrix(y_test, y_pred))

    # Calculate average metrics and log to wandb
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average Metrics Across Groups:", avg_test_metrics)

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
    if df.empty:
        raise ValueError("create_norm_pca_df: Input DataFrame is empty.")
    participant_frames_labels = df.iloc[:, :4]
    
    features = df.columns[4:]
    norm_df = df.copy()
    
    scaler = StandardScaler()
    norm_df[features] = scaler.fit_transform(norm_df[features])
    
    norm_df = pd.concat([participant_frames_labels, norm_df[features]], axis=1)

    x = df.iloc[:, 4:]
    x = StandardScaler().fit_transform(x.values)

    pca = PCA(n_components=0.90)
    principal_components = pca.fit_transform(x)
    print(principal_components.shape)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
    principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

    return principal_df    

def main():
    sweep_config = {
        'method': 'random',
        'name': 'brirl_linear_inter',
        'parameters': {
            'binary_multiclass': {'values': ['binary', 'multiclass']},
            'feature_set': {'values': ["full", "stats", "rf"]},
            'dataset': {'values': ['reg', 'norm', 'pca']},
            'alpha': {'values': [0.00001, 0.0001, 0.001, 0.01, 0.1]},
            'max_iter': {'values': [100, 500, 1000, 2000]},
            'tol': {'values': [1e-4, 1e-3, 1e-2]},
            'sequence_length': {'values': [30, 60, 90, 150, 300]},
            'modality': {'values': [
                'pose', 'facial', 'audio', 'text',
                'pose_facial', 'pose_audio', 'pose_text',
                'facial_audio', 'facial_text',
                'audio_text',
                'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
                'facial_audio_text',
                'pose_facial_audio_text',
            ]},
            'model_type': {'values': ['sgd']}
        }
    }

    print(sweep_config)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_linear_inter")
    wandb.agent(sweep_id, function=train)


if __name__ == '__main__':
    main()