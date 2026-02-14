import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import wandb
from itertools import product
from get_all_metrics import get_all_metrics
from create_data_splits import make_split_indices


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
    
    metrics_all = {k: [] for k in [
        "test_accuracy", "test_precision", "test_recall", "test_f1",
        "test_accuracy_tolerant", "test_precision_tolerant",
        "test_recall_tolerant", "test_f1_tolerant",
        "test_auc", "test_fnr",
        "test_windowed_accuracy", "test_windowed_precision",
        "test_windowed_recall", "test_windowed_f1",
        "test_earliest_detection_time",
    ]}

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
        

        # balance training
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        knn = KNeighborsClassifier(n_neighbors=config.n_neighbors)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)

        #wandb.log({f"{pid}_probs": wandb.Histogram(y_proba)})

        m = get_all_metrics(y_pred, y_test, y_pred_proba=y_proba,
                            sessions=np.full(len(y_test), pid),
                            window_size=None, tolerance=1)
        m.pop("test_edt_per_session", None)
        for k, v in m.items():
            if k in metrics_all:
                metrics_all[k].append(v)
        wandb.log({f"{pid}_{k}": v for k, v in m.items()})
        print(pid, m)
        print(confusion_matrix(y_test, y_pred))

    # aggregate across participants
    avg = {f"avg_{k}": np.mean(v) for k,v in metrics_all.items()}
    print("Overall:", avg)
    wandb.log(avg)


def validate_modality_feature_combination(modality, feature_set):
    modality_components = modality.split('_')
    
    if 'text' in modality_components and feature_set != 'full':
        return False
        
    if feature_set == 'stats':
        valid_components = ['facial', 'audio']
        for component in modality_components:
            if component not in valid_components:
                return False
    
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
    # A single sweep with all possible feature sets
    sweep_config = {
        'method': 'random',
        'name': 'brirl_linear_intra',
        'parameters': {
            'split_strategy': {'values': ['binary', 'multiclass', 'multiclass_exclude_neutral','multiclass_to_binary']},
            'feature_set': {'values': ['full', 'stats', 'rf']},
            'dataset': {'values': ['reg', 'norm', 'pca']},
            'n_neighbors': {'values': [3, 5, 7, 10, 15]},
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
            'model_type': {'values': ['knn']}
        }
    }
    
    print(sweep_config)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_linear_intra")
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    main()
    