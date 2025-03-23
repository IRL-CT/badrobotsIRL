import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import wandb
from itertools import product
from get_metrics import get_test_metrics
from create_data_splits import create_data_splits
import random

def train():

    wandb.init()
    config = wandb.config
    print(config)
    seed_value = 42

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

    # select dataset and modalities
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

    modality_components = config.modality.split('_')
    selected_modalities = {}

    feature_set = config.feature_set

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
    fold_importances = []

        
    for fold in range(5):
        splits = create_data_splits(
            df, "multiclass",
            fold_no=fold,
            num_folds=5,
            seed_value=42,
            sequence_length=config.sequence_length)
        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits
            
        feature_names = X_train.columns
        
        # balance training dataset
        smote = SMOTE(random_state=42) 
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        rf = RandomForestClassifier(
                random_state=seed_value,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth
            )
        rf.fit(X_train_balanced, y_train_balanced)
    
        y_pred = rf.predict(X_test)

        # logging prediction probabilities
        y_pred_proba = rf.predict_proba(X_test)

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

        # wandb.log({f"t{fold}_conf_mat" : wandb.plot.confusion_matrix(probs=None,
        #         y_true=y_test.astype(int) , preds=y_pred.astype(int) ,
        #         class_names=['no_discomfort', 'is_discomfort'])})
        
        # print(test_metrics)
        print(confusion_matrix(y_test, y_pred))
        print(f'Fold {fold} Feature Importance:{rf.feature_importances_}')
        fold_importances.append(rf.feature_importances_)

    # Calculate average metrics and log to wandb
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average Metrics Across Groups:", avg_test_metrics)

    avg_feature_importances = np.mean(fold_importances, axis=0)
    feature_importance_dict = {feature_names[i]: avg_feature_importances[i] for i in range(len(feature_names))}
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    print("Sorted Feature Importances:", sorted_feature_importance)
    # wandb.log({"feature_importances": feature_importance_dict})


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
    # global df
    # global df_norm
    # global df_pca 
    # df = pd.read_csv("../preprocessing/merged_features/all_participants_0_3.csv")
    # df_stats = pd.read_csv("../preprocessing/stats_features/all_participants_merged_correct_stats_0_3.csv")
    # df_norm = create_normalized_df(df)
    # df_pca = create_norm_pca_df(df)

    # Sweep configuration

    feature_set = random.choice(["full", "stats", "rf"])

    if feature_set == "full":
        modality = random.choice(
        ['pose', 'facial', 'audio', 'text',
         'pose_facial', 'pose_audio', 'pose_text',
         'facial_audio', 'facial_text',
         'audio_text',
         'pose_facial_audio', 'pose_facial_text', 'pose_audio_text',
         'facial_audio_text',
         'pose_facial_audio_text',
         ])
    elif feature_set == "stats":
        modality = random.choice(['facial_audio', 'facial', 'audio'])
    elif feature_set == "rf":
        modality = random.choice(['pose_facial_audio', 'pose', 'facial', 'audio', 'pose_facial', 'pose_audio', 'facial_audio'])

    sweep_config = {
        'method': 'random',
        'name': 'random_forest_tuning',
        'parameters': {
            'feature_set': {'values': [feature_set]}, # Full, Stat, RF, Quali
            'dataset': {'values': ['reg', 'normalized', 'pca']},
            'n_estimators': {'values': [100, 200, 300, 500, 700, 1000]},
            'max_depth': {'values': [5, 10, 15, 20, 25, 30]},
            'modality': {'values': [modality]},
            'sequence_length' : {'values': [30, 60, 90, 150, 300]}
        }
    }
        
    print(sweep_config)
    
    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="rf_multiclass_v1")
    wandb.agent(sweep_id, function=train)



if __name__ == '__main__':
    main()