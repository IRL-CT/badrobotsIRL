import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import wandb
from get_metrics import get_test_metrics

def train():
    wandb.init()
    config = wandb.config
    print(config)
    
    seed_value = 42

    if not validate_modality_feature_combination(config.modality, config.feature_set):
        print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
        wandb.log({"status": "skipped_invalid_combination"})
        return

    # Load datasets
    df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")
    df_text = pd.read_csv("../../preprocessing/clip_text_embeddings.csv")

    info = df.iloc[:, :4]
    df_pose_index = df.iloc[:, 4:28]
    df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
    df_audio_index = df.iloc[:, 63:88]
    df_text_index = df_text.iloc[:, 2:]

    df_facial_index_stats = df_stats.iloc[:, 4:30]
    df_audio_index_stats = df_stats.iloc[:, 30:53]

    df_facial_index_rf = df_rf.iloc[:, 38:]
    df_pose_index_rf = df_rf.iloc[:, 4:28]
    df_audio_index_rf = df_rf.iloc[:, 28:38]

    modalities = {
        "pose_full": df_pose_index, "pose_rf": df_pose_index_rf,
        "facial_full": df_facial_index, "facial_stats": df_facial_index_stats, "facial_rf": df_facial_index_rf,
        "audio_full": df_audio_index, "audio_stats": df_audio_index_stats, "audio_rf": df_audio_index_rf,
        "text_full": df_text_index
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

    print(df.shape)

    test_metrics_list = {
        "test_accuracy": [], "test_precision": [], "test_recall": [], "test_f1": [],
        "test_accuracy_tolerant": [], "test_precision_tolerant": [],
        "test_recall_tolerant": [], "test_f1_tolerant": []
    }

    # Intra-participant training
    for pid in sorted(df["participant"].unique()):
        d = df[df["participant"] == pid]
        if d.empty:
            continue

        X = d.iloc[:, 4:]
        y = d["binary_label"] if config.binary_multiclass == "binary" else d["multiclass_label"]

        # 70/10/20 train/val/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=seed_value, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=2/3, random_state=seed_value, stratify=y_temp
        )

        # Balance training set
        X_train_balanced, y_train_balanced = SMOTE(random_state=seed_value).fit_resample(X_train, y_train)

        # SVC model
        svm = SVC(
            C=config.C if hasattr(config, 'C') else 1.0,
            kernel=config.kernel if hasattr(config, 'kernel') else 'rbf',
            gamma=config.gamma if hasattr(config, 'gamma') else 'scale',
            probability=True,
            random_state=seed_value
        )
        svm.fit(X_train_balanced, y_train_balanced)

        y_pred = svm.predict(X_test)
        y_proba = svm.predict_proba(X_test)

        # W&B logging
        df_probs = pd.DataFrame(y_proba, columns=[f"prob_class_{i}" for i in range(y_proba.shape[1])])
        df_probs["y_pred"] = y_pred
        df_probs["y_true"] = y_test
        wandb.log({f"{pid}_probs": wandb.Histogram(y_proba)})

        m = get_test_metrics(y_pred, y_test, tolerance=1)
        for k, v in m.items():
            test_metrics_list[k].append(v)
        wandb.log({f"{pid}_{k}": v for k, v in m.items()})

        print(f"Participant {pid} metrics:", m)
        print(f"Participant {pid} confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Average metrics
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)
    print("Average metrics across participants:", avg_test_metrics)



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
        'name': 'svm_intraparticipant_tuning',
        'parameters': {
            'binary_multiclass': {'values': ['binary', 'multiclass']},
            'feature_set': {'values': ["full", "stats", "rf"]},
            'dataset': {'values': ['reg', 'norm', 'pca']},
            'C': {'values': [0.1, 1, 10, 100]},
            'kernel': {'values': ['linear', 'rbf', 'poly']},
            'gamma': {'values': ['scale', 'auto']},
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
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="svm_intraparticipant")
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    main()
