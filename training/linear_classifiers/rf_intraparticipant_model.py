import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import wandb
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from get_metrics import get_test_metrics
from create_data_splits import make_split_indices

def train():
    wandb.init()
    config = wandb.config
    print(config)
    
    # Validate modality/feature_set combination
    if not validate_modality_feature_combination(config.modality, config.feature_set):
        print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
        wandb.log({"status": "skipped_invalid_combination"})
        return
    
    seed_value = 42

    # Load data
    df = pd.read_csv("../../data/raw/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../data/feature_sets/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../data/feature_sets/all_participants_rf_0_3_40.csv")
    df_text = pd.read_csv("../../data/embeddings/clip_text_embeddings_pca.csv")
    df_text_pca = pd.read_csv("../../data/embeddings/clip_text_embeddings_pca.csv")

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

    # Select dataset/modalities
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

    # Normalize or PCA
    if config.dataset == "norm":
        df = create_normalized_df(df)
    elif config.dataset == "pca":
        df = create_norm_pca_df(df)

    print(df.shape)

    # Metrics storage
    test_metrics_list = {
        "test_accuracy": [], "test_precision": [], "test_recall": [], "test_f1": [],
        "test_accuracy_tolerant": [], "test_precision_tolerant": [],
        "test_recall_tolerant": [], "test_f1_tolerant": []
    }
    fold_importances = []

    # Loop over participants
    for pid in sorted(df["participant"].unique()):
        d = df[df["participant"] == pid]
        if d.empty:
            continue

        #X = d.iloc[:, 4:]
        #y = d["binary_label"] if config.binary_multiclass == "binary" else d["multiclass_label"]

        train_indices, train_labels, test_indices, test_labels = make_split_indices(
            d, config.split_strategy, seed=seed_value
        )

        features = d.iloc[:, 4:].values
        #labels = d[label_column].values.astype(int)

        X_train_all = features[train_indices]
        y_train_all = train_labels
        X_test = features[test_indices]
        y_test = test_labels

        # first split train vs (val+test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, train_size=config.train_ratio, random_state=42, stratify=y_train_all
        )

        #concatenate val to train
        

        

        # balance training
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)


        # Random Forest
        rf = RandomForestClassifier(
            random_state=seed_value,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)

        # W&B logging
        df_probs = pd.DataFrame(y_proba, columns=[f"prob_class_{i}" for i in range(y_proba.shape[1])])
        df_probs["y_pred"] = y_pred
        df_probs["y_true"] = y_test
        #wandb.log({f"{pid}_probs": wandb.Histogram(y_proba)})

        m = get_test_metrics(y_pred, y_test, tolerance=1)
        for k, v in m.items():
            test_metrics_list[k].append(v)
        wandb.log({f"{pid}_{k}": v for k, v in m.items()})
        print(f"Participant {pid} metrics:", m)

        fold_importances.append(rf.feature_importances_)
        print(f"Participant {pid} confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Aggregate metrics
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)
    print("Average metrics across participants:", avg_test_metrics)

    # Aggregate feature importances
    feature_names = X.columns
    avg_feature_importances = np.mean(fold_importances, axis=0)
    feature_importance_dict = {feature_names[i]: avg_feature_importances[i] for i in range(len(feature_names))}
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    print("Sorted feature importances:", sorted_feature_importance)


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
        'name': 'brirl_linear_intra',
        'parameters': {
            'split_strategy': {'values': ['binary', 'multiclass', 'multiclass_exclude_neutral','multiclass_to_binary']},
            'feature_set': {'values': ['full', 'stats', 'rf']},
            'dataset': {'values': ['reg', 'norm', 'pca']},
            'n_estimators': {'values': [100, 200, 300, 500, 700, 1000]},
            'max_depth': {'values': [5, 10, 15, 20, 25, 30]},
            'sequence_length' : {'values': [5,10, 30, 60, 90, 150]},
            'modality': {'values': [
                'pose', 'facial', 'audio', 'text',
                'pose_facial', 'pose_audio', 'pose_text',
                'facial_audio', 'facial_text',
                'audio_text',
                'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
                'facial_audio_text',
                'pose_facial_audio_text',
            ]},
            'train_ratio': {'values': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
            'model_type': {'values': ['rf']}
        }
    }
        
    print(sweep_config)
    
    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_linear_intra")
    wandb.agent(sweep_id, function=train)



if __name__ == '__main__':
    main()