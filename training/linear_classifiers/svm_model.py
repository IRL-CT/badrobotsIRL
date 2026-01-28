import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb
import random
from get_metrics import get_test_metrics
from create_data_splits import create_data_splits

def train():
    wandb.init()
    config = wandb.config
    print(config)
    
    # Skip validation for curated feature sets
    if config.feature_set not in ["curated_v0", "curated_v1", "curated_v3"]:
        # Validate modality and feature_set combination
        is_valid_combination = validate_modality_feature_combination(config.modality, config.feature_set)
        
        if not is_valid_combination:
            print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
            # Log that this was skipped
            wandb.log({"status": "skipped_invalid_combination"})
            return
    
    seed_value = 42

    # Handle curated feature sets
    if config.feature_set == "curated_v0":
        df = pd.read_csv("../../preprocessing/curated_features/curated_features_dataset_v0.csv")
        if config.dataset == "norm":
            df = create_normalized_df(df)
        elif config.dataset == "pca":
            df = create_norm_pca_df(create_normalized_df(df))
    elif config.feature_set == "curated_v1":
        df = pd.read_csv("../../preprocessing/curated_features/curated_features_dataset_v1.csv")
        if config.dataset == "norm":
            df = create_normalized_df(df)
        elif config.dataset == "pca":
            df = create_norm_pca_df(create_normalized_df(df))
    elif config.feature_set == "curated_v3":
        df = pd.read_csv("../../preprocessing/curated_features/curated_features_dataset_v3.csv")
        if config.dataset == "norm":
            df = create_normalized_df(df)
        elif config.dataset == "pca":
            df = create_norm_pca_df(create_normalized_df(df))
    else:
        # Original feature set loading
        df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
        df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
        df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")
        df_text = pd.read_csv("../../preprocessing/clip_text_embeddings_pca.csv")
        df_text_pca = pd.read_csv("../../preprocessing/clip_text_embeddings_pca.csv")
        df_text_distance = pd.read_csv("../../preprocessing/clip_text_cosine_similarity.csv")

        info = df.iloc[:, :4]
        df_pose_index = df.iloc[:, 4:28]
        df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1) # action units, gaze
        df_audio_index = df.iloc[:, 63:88]
        df_text_index = df_text.iloc[:, 2:]
        df_text_distance = df_text_distance.iloc[:, 2:]

        df_facial_index_stats = df_stats.iloc[:, 4:30]
        df_audio_index_stats = df_stats.iloc[:, 30:53]
        df_all_stats = df_stats.iloc[:, 4:]

        df_facial_index_rf = df_rf.iloc[:, 38:]
        df_pose_index_rf = df_rf.iloc[:, 4:28]
        df_audio_index_rf = df_rf.iloc[:, 28:38]
        df_all_rf = df_rf.iloc[:, 4:]

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
            "all_stats": df_all_stats,
            "all_rf": df_all_rf,
            "text_distance": df_text_distance
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
        if "cosine" in modality_components:
            selected_modalities["text_distance"] = modalities["text_distance"]

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
            seed_value=seed_value)
        X_train, X_val, X_test, y_train, y_val, y_test, _, _, _, _, _, _, _ = splits
        
        smote = SMOTE(random_state=seed_value) 
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        svm = SVC(
            C=config.C if hasattr(config, 'C') else 1.0,
            kernel=config.kernel if hasattr(config, 'kernel') else 'rbf',
            gamma=config.gamma if hasattr(config, 'gamma') else 'scale',
            probability=True,
            random_state=seed_value
        )

        svm.fit(X_train_balanced, y_train_balanced)
        y_pred = svm.predict(X_test)
        y_pred_proba = svm.predict_proba(X_test)

        df_probs = pd.DataFrame(y_pred_proba, columns=[f"prob_class_{i}" for i in range(y_pred_proba.shape[1])])
        df_probs["y_pred"] = y_pred
        df_probs["y_true"] = y_test
        
        wandb.log({f"fold_{fold}_prediction_probabilities": wandb.Histogram(y_pred_proba)})
        
        test_metrics = get_test_metrics(y_pred, y_test, tolerance=1)
        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        wandb.log({f"t{fold}_{k}": v for k, v in test_metrics.items()})

        print(confusion_matrix(y_test, y_pred))

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
            'feature_set': {'values': ['full', 'stats', 'rf', 'curated_v3', 'curated_v1']},
            'dataset': {'values': ['reg', 'norm', 'pca']},
            'C': {'values': [0.1, 1, 10, 100]},
            'kernel': {'values': ['linear', 'rbf', 'poly']},
            'gamma': {'values': ['scale', 'auto']},
            'sequence_length': {'values': [30, 60, 90, 150, 300]},
            'modality': {'values': [
                'pose', 'facial', 'audio', 'text', 'cosine',
                'pose_facial', 'pose_audio', 'pose_text', 'pose_cosine', 
                'facial_audio', 'facial_text', 'facial_cosine',
                'audio_text', 'audio_cosine',
                'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
                'facial_audio_text', 'facial_audio_cosine', 'pose_facial_audio_cosine',
                'pose_facial_audio_text', 'pose_audio_cosine', 

            ]},
            'model_type': {'values': ['svm']}
        }
    }
    
    print(sweep_config)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_linear_inter")
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    main()
