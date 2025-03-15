import wandb
import random
import numpy as np
import tensorflow as tf
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from create_data_splits import create_data_splits, create_data_splits_pca
from get_metrics import get_test_metrics

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

def baseline(df):

    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }

    for fold in range(5):

        splits = create_data_splits(df, "multiclass", fold_no=fold, num_folds=5, seed_value=42, sequence_length=10)

        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

        majority_class = Counter(y_train).most_common(1)[0][0]

        y_pred = [majority_class] * len(y_test)

        # model = DummyClassifier(strategy="most_frequent")
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        test_metrics = {"test_accuracy": accuracy, 
                        "test_precision": precision,
                        "test_recall": recall,
                        "test_f1": f1}
        
        print("Fold", fold, "test metrics", test_metrics)

        wandb.log({f"fold_{fold}_metrics": test_metrics})

        test_metrics_list["test_accuracy"].append(accuracy)
        test_metrics_list["test_precision"].append(precision)
        test_metrics_list["test_recall"].append(recall)
        test_metrics_list["test_f1"].append(f1)
    
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)
    print("Average Test Metrics Across All Folds:", avg_test_metrics)

    return test_metrics_list


def train():

    wandb.init()
    config = wandb.config
    print(config)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

    feature_set = config.feature_set
    data = config.data

    df_full = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
    df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")

    if feature_set == "full":
        df = df_full
    elif feature_set == "rf":
        df = df_rf
    elif feature_set == "stats":
        df = df_stats

    if data == "norm":
        df = create_normalized_df(df)
    elif data == "pca":
        df = create_pca_df(df)

    baseline(df)


def main():

    sweep_config = {
        'method': 'random',
        'name': f'baseline_multiclass',
        'parameters': {
            'feature_set' : {'values': ["full", "rf", "stats"]},
            'data' : {'values' : ["reg", "norm", "pca"]},
        }
        # feature set (full, stats, rf) -> modality selection (combined, pose, facial, etc.) -> (reg, norm, pca) -> fusion
    }

    print(sweep_config)
    
    def train_wrapper():
        train()

    sweep_id = wandb.sweep(sweep=sweep_config, project=f"baseline_multiclass")
    wandb.agent(sweep_id, function=train_wrapper)


if __name__ == '__main__':
    main()

    # df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
    # df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")

    # info = df.iloc[:, :4]
    # df_pose_index = df.iloc[:, 4:28]
    # df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
    # df_audio_index = df.iloc[:, 63:88]

    # df_facial_index_stats = df_stats.iloc[:, 4:30]
    # df_audio_index_stats = df_stats.iloc[:, 30:53]

    # df_facial_index_rf = df_rf.iloc[:, 38:]
    # df_pose_index_rf = df_rf.iloc[:, 4:28]
    # df_audio_index_rf = df_rf.iloc[:, 28:38]

    # modality_mapping = {
    #     "pose": pd.concat([info, df_pose_index], axis=1),
    #     "facial": pd.concat([info, df_facial_index], axis=1),
    #     "audio": pd.concat([info, df_audio_index], axis=1)
    # }

    # print(create_normalized_df(modality_mapping.get("pose")))