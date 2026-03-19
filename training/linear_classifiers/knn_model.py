import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import wandb
from itertools import product
from get_all_metrics import get_all_metrics
from create_data_splits import create_data_splits
from registry_utils import append_to_registry


# Feature sets that use the full dataset without modality selection
NO_MODALITY_SELECTION_SETS = {"catch22", "tsfresh", "curated_v4", "rf", "selectkbest", "curated_v5"}


def apply_aux_windowing(df_aux, participant_col_idx, feature_start_col, window_size, stride, aggregation):
    """Apply windowing to an auxiliary DataFrame (e.g. text embeddings, cosine similarity).

    Groups by participant, slides windows of size `window_size` with `stride`,
    and aggregates feature columns using the same aggregation as the main DF.

    Returns a DataFrame of aggregated feature columns only (equivalent to .iloc[:, feature_start_col:]).
    """
    df_aux = df_aux.reset_index(drop=True)
    participant_col = df_aux.columns[participant_col_idx]
    feature_cols = df_aux.columns[feature_start_col:].tolist()

    rows = []
    for participant in df_aux[participant_col].unique():
        p_positions = np.where(df_aux[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            window = df_aux.iloc[window_pos]
            if aggregation == "average":
                row = window[feature_cols].mean().to_dict()
            elif aggregation == "mode":
                row = window[feature_cols].mode().iloc[0].to_dict()
            else:  # 'last'
                row = window[feature_cols].iloc[-1].to_dict()
            rows.append(row)
            start += stride

    return pd.DataFrame(rows, columns=feature_cols).reset_index(drop=True)


def apply_windowing(df, window_size, stride, aggregation):
    """Aggregate a per-frame DataFrame into windowed samples.

    Windowing is applied per participant (column 0). Features (columns 4+)
    are aggregated with the specified method. Labels follow:
      - 'average' / 'last'  → last frame of the window
      - 'mode'              → statistical mode of the window

    Returns
    -------
    windowed_df : pd.DataFrame
        New DataFrame with one row per window.
    last_positions : list[int]
        Positional indices (in the reset-index view of df) of the last frame
        of each window — used to align auxiliary DataFrames (e.g. text embeddings).
    """
    df_reset = df.reset_index(drop=True)
    participant_col = df_reset.columns[1]
    info_cols = df_reset.columns[:4].tolist()
    feature_cols = df_reset.columns[4:].tolist()
    binary_col = info_cols[2]
    multiclass_col = info_cols[3]

    rows = []
    last_positions = []

    for participant in df_reset[participant_col].unique():
        p_positions = np.where(df_reset[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            last_pos = int(p_positions[end - 1])
            window = df_reset.iloc[window_pos]
            last_row = df_reset.iloc[last_pos]
            last_positions.append(last_pos)

            row = {
                info_cols[0]: last_row[info_cols[0]],
                info_cols[1]: last_row[info_cols[1]],
            }

            # Labels
            if aggregation == "mode":
                row[binary_col] = window[binary_col].mode().iloc[0]
                row[multiclass_col] = window[multiclass_col].mode().iloc[0]
            else:  # 'average' or 'last'
                row[binary_col] = last_row[binary_col]
                row[multiclass_col] = last_row[multiclass_col]

            # Features
            if aggregation == "average":
                for col in feature_cols:
                    row[col] = window[col].mean()
            elif aggregation == "mode":
                for col in feature_cols:
                    row[col] = window[col].mode().iloc[0]
            else:  # 'last'
                for col in feature_cols:
                    row[col] = last_row[col]

            rows.append(row)
            start += stride

    windowed_df = pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)
    windowed_df = windowed_df.dropna(axis=1, how="any").dropna(axis=0, how="any").reset_index(drop=True)

    print(f"Applied windowing: window_size={window_size}, stride={stride}, aggregation={aggregation}")
    print(f"Original samples: {len(df)}, Windowed samples: {len(windowed_df)}")
    return windowed_df, last_positions


def apply_catch22_windowing(df, window_size, stride):
    """Apply catch22 feature extraction on sliding windows per participant.

    For each window, catch22 computes 22 time-series features per input
    feature column (e.g. col__DN_HistogramMode_5, ...).
    Labels use the last frame of each window.

    Returns
    -------
    pd.DataFrame with 4 info columns + (n_features * 22) catch22 columns.
    """
    import pycatch22

    df_reset = df.reset_index(drop=True)
    participant_col = df_reset.columns[1]
    info_cols = df_reset.columns[:4].tolist()
    feature_cols = df_reset.columns[4:].tolist()
    binary_col = info_cols[2]
    multiclass_col = info_cols[3]

    rows = []
    for participant in df_reset[participant_col].unique():
        p_positions = np.where(df_reset[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            window = df_reset.iloc[window_pos]
            last_row = df_reset.iloc[int(p_positions[end - 1])]

            row = {
                info_cols[0]: last_row[info_cols[0]],
                info_cols[1]: last_row[info_cols[1]],
                binary_col: last_row[binary_col],
                multiclass_col: last_row[multiclass_col],
            }

            for col in feature_cols:
                series = window[col].values.tolist()
                result = pycatch22.catch22_all(series)
                for name, val in zip(result["names"], result["values"]):
                    row[f"{col}__{name}"] = val

            rows.append(row)
            start += stride

    result = pd.DataFrame(rows).reset_index(drop=True)
    result = result.dropna(axis=1, how="any").dropna(axis=0, how="any").reset_index(drop=True)
    return result


def apply_tsfresh_windowing(df, window_size, stride):
    """Apply tsfresh feature extraction on sliding windows per participant.

    Constructs a long-format DataFrame for all windows, then calls
    tsfresh.extract_features() in a single pass for efficiency.
    Labels use the last frame of each window. NaN feature columns are dropped.

    Returns
    -------
    pd.DataFrame with 4 info columns + tsfresh-extracted feature columns.
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters

    df_reset = df.reset_index(drop=True)
    participant_col = df_reset.columns[1]
    info_cols = df_reset.columns[:4].tolist()
    feature_cols = df_reset.columns[4:].tolist()
    binary_col = info_cols[2]
    multiclass_col = info_cols[3]

    info_rows = []
    window_slices = []
    window_id = 0

    for participant in df_reset[participant_col].unique():
        p_positions = np.where(df_reset[participant_col] == participant)[0]
        n = len(p_positions)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_pos = p_positions[start:end]
            window = df_reset.iloc[window_pos]
            last_row = df_reset.iloc[int(p_positions[end - 1])]

            info_rows.append({
                info_cols[0]: last_row[info_cols[0]],
                info_cols[1]: last_row[info_cols[1]],
                binary_col: last_row[binary_col],
                multiclass_col: last_row[multiclass_col],
                "_window_id": window_id,
            })

            slice_df = window[feature_cols].copy()
            slice_df = slice_df.reset_index(drop=True)
            slice_df["_window_id"] = window_id
            slice_df["_time"] = range(window_size)
            window_slices.append(slice_df)

            window_id += 1
            start += stride

    tsfresh_input = pd.concat(window_slices, ignore_index=True)
    extracted = extract_features(
        tsfresh_input,
        column_id="_window_id",
        column_sort="_time",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True,
    )
    # Drop columns with any NaN (some tsfresh features fail on short windows)
    extracted = extracted.dropna(axis=1, how="any").reset_index(drop=True)

    info_df = pd.DataFrame(info_rows).drop(columns=["_window_id"]).reset_index(drop=True)
    result = pd.concat([info_df, extracted], axis=1)
    return result.dropna(axis=0, how="any").reset_index(drop=True)


def train():
    wandb.init()
    config = wandb.config
    print(config)

    # Skip modality validation for feature sets that ignore modality selection
    if config.feature_set not in NO_MODALITY_SELECTION_SETS:
        is_valid_combination = validate_modality_feature_combination(config.modality, config.feature_set)
        if not is_valid_combination:
            print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
            wandb.log({"status": "skipped_invalid_combination"})
            return

    seed_value = 42

    # ------------------------------------------------------------------
    # Load base 100fps dataset
    # ------------------------------------------------------------------
    if config.feature_set in ["curated_v4"]: #includes 'catch22', 'tsfresh'
        df_base = pd.read_csv("../../preprocessing/curated_features/curated_features_dataset_v4.csv")
    elif config.feature_set == "curated_v5":
        df_base = pd.read_csv("../../preprocessing/curated_features/curated_features_dataset_v5.csv")
    elif config.feature_set == "rf":
        df_base = pd.read_csv("../../preprocessing/rf_features/allparticipants_rf_100fps.csv")
    elif config.feature_set == "selectkbest":
        if config.binary_multiclass == "binary":
            df_base = pd.read_csv("../../preprocessing/interpolation/select_k_best_allparticipants_100fps_binary_label.csv")
        else:
            df_base = pd.read_csv("../../preprocessing/interpolation/select_k_best_allparticipants_100fps_multiclass_label.csv")
    elif config.feature_set in ["full"]:
        # 'full' — all start from the full 100fps feature set
        df_base = pd.read_csv("../../preprocessing/full_features/allparticipants_100fps.csv")
        #see unique participants
        print("Unique participants in base dataset:", df_base.iloc[:, 1].unique())
    # ------------------------------------------------------------------
    # Apply windowing / on-the-fly feature extraction
    # ------------------------------------------------------------------
    window = config.window
    win_stride = min(config.stride, window)
    wandb.log({"effective_stride": win_stride, "effective_window": window})

    if config.feature_set == "catch22":
        df = apply_catch22_windowing(df_base, window, win_stride)
        last_positions = None
    elif config.feature_set == "tsfresh":
        df = apply_tsfresh_windowing(df_base, window, win_stride)
        last_positions = None
    else:
        df, last_positions = apply_windowing(df_base, window, win_stride, config.aggregation)

    # ------------------------------------------------------------------
    # Feature / modality selection (only for 'full' feature set)
    # ------------------------------------------------------------------
    if config.feature_set not in NO_MODALITY_SELECTION_SETS:
        info = df.iloc[:, :4]
        if config.feature_set == "selectkbest":
            # Name-based modality splitting for selectkbest
            feature_cols = df.columns[4:]
            facial_cols = [c for c in feature_cols if c.startswith('AU') or c.startswith('gaze_')]
            audio_cols = [c for c in feature_cols if c not in facial_cols]
            df_pose_index = pd.DataFrame(index=df.index)  # no pose features
            df_facial_index = df[facial_cols]
            df_audio_index = df[audio_cols]
        else:
            df_pose_index = df.iloc[:, 4:28]
            df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
            df_audio_index = df.iloc[:, 63:88]

        modality_components = config.modality.split("_")

        # Text embeddings must be aligned to the windowed rows via last_positions
        if "text" in modality_components or "cosine" in modality_components:
            df_text_raw = pd.read_csv("../../preprocessing/clip_text_embeddings_pca.csv")
            df_text_cos_raw = pd.read_csv("../../preprocessing/clip_text_cosine_similarity.csv")
            df_text_index = df_text_raw.iloc[last_positions, 2:].reset_index(drop=True)
            df_text_distance = df_text_cos_raw.iloc[last_positions, 2:].reset_index(drop=True)

        selected_modalities = {}
        if "pose" in modality_components:
            selected_modalities["pose_full"] = df_pose_index
        if "facial" in modality_components:
            selected_modalities["facial_full"] = df_facial_index
        if "audio" in modality_components:
            selected_modalities["audio_full"] = df_audio_index
        if "text" in modality_components:
            selected_modalities["text_full"] = df_text_index
        if "cosine" in modality_components:
            selected_modalities["text_distance"] = df_text_distance

        df = info
        for m in selected_modalities.values():
            df = pd.concat([df, m.reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------------
    # Normalization / PCA
    # ------------------------------------------------------------------
    if config.dataset == "norm":
        df = create_normalized_df(df)
    elif config.dataset == "pca":
        df = create_norm_pca_df(df)

    print(df)
    print(df.shape)

    # ------------------------------------------------------------------
    # Feature randomization (if enabled)
    # ------------------------------------------------------------------
    #do this onlyy for curated dataset
    if config.feature_randomizer == 1 and config.feature_set in ["curated_v4", "curated_v5"]:
        available_features = df.columns[4:].tolist()
        max_features = len(available_features)

        if max_features > 0:
            n_features = np.random.randint(1, max_features + 1)
            selected_features = np.random.choice(available_features, size=n_features, replace=False).tolist()
            df = df[df.columns[:4].tolist() + selected_features]

            wandb.log({
                "features_included": selected_features,
                "n_features_selected": n_features,
                "total_available_features": max_features,
            })
            print(f"Feature randomization enabled: Selected {n_features} out of {max_features} features")
            print(f"Selected features: {selected_features}")
        else:
            print("Warning: No features available for randomization")
            wandb.log({"features_included": [], "n_features_selected": 0, "total_available_features": 0})
    else:
        all_features = df.columns[4:].tolist()
        wandb.log({
            "features_included": all_features,
            "n_features_selected": len(all_features),
            "total_available_features": len(all_features),
        })
        print(f"Feature randomization disabled: Using all {len(all_features)} features")

    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": [],
        "test_auc": [],
        "test_fnr": [],
        "test_windowed_accuracy": [],
        "test_windowed_precision": [],
        "test_windowed_recall": [],
        "test_windowed_f1": [],
        "test_earliest_detection_time": [],
    }

    all_fold_predictions = []  # collected across folds for artifact export

    for fold in range(5):
        splits = create_data_splits(
            df, config.binary_multiclass,
            fold_no=fold,
            num_folds=5,
            seed_value=42,
            sequence_length=1)
        X_train, X_val, X_test, y_train, y_val, y_test, _, _, _, _, _, _, _, session_train, session_val, session_test = splits

        # Balance training dataset
        min_class_count = min(np.bincount(y_train))
        smote_k = min(5, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=smote_k)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=config.n_neighbors)
        knn.fit(X_train_balanced, y_train_balanced)

        y_pred = knn.predict(X_test)

        # Logging prediction probabilities
        y_pred_proba = knn.predict_proba(X_test)

        df_probs = pd.DataFrame(y_pred_proba, columns=[f"prob_class_{i}" for i in range(y_pred_proba.shape[1])])
        df_probs["y_pred"] = y_pred
        df_probs["y_true"] = y_test

        effective_test_stride = min(config.test_stride, config.test_window_size)
        wandb.log({"effective_test_stride": effective_test_stride, "effective_test_window_size": config.test_window_size})
        test_metrics = get_all_metrics(y_pred, y_test, y_pred_proba=y_pred_proba,
                                       sessions=session_test, window_size=config.test_window_size,
                                       stride=effective_test_stride, tolerance=1)
        test_metrics.pop("test_edt_per_session", None)
        for key in test_metrics:
            if key in test_metrics_list:
                test_metrics_list[key].append(test_metrics[key])
        wandb.log({f"t{fold}_{k}": v for k, v in test_metrics.items()})
        print(test_metrics)

        print(confusion_matrix(y_test, y_pred))

        # Collect predictions for ensemble artifact
        fold_df = pd.DataFrame({
            "fold": fold,
            "y_pred": y_pred,
            "y_true": y_test,
            "session": session_test,
        })
        all_fold_predictions.append(fold_df)

    # Register this run (with predictions embedded) for ensemble discovery
    append_to_registry(wandb.run.id, dict(config), all_fold_predictions)

    # Calculate average metrics and log to wandb
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average Metrics Across Groups:", avg_test_metrics)


def validate_modality_feature_combination(modality, feature_set):
    modality_components = modality.split("_")

    # Text only works with 'full' feature set
    if "text" in modality_components and feature_set != "full":
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

    principal_df = pd.DataFrame(data=principal_components, columns=["principal component " + str(i) for i in range(principal_components.shape[1])])
    principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

    return principal_df


def main():
    sweep_config = {
        "method": "random",
        "name": "brirl_linear_inter_2",
        "parameters": {
            "binary_multiclass": {"values": ["binary", "multiclass"]},
            "feature_set": {"values": ["full", "rf", "catch22", "tsfresh", "curated_v4","selectkbest", "curated_v5"]},
            "dataset": {"values": ["reg", "norm", "pca"]},
            # Dataset-level windowing (applied to 100fps data before training)
            "window": {"values": [1, 5, 10, 15, 30]},
            "stride": {"values": [1, 3, 5, 10]},
            "aggregation": {"values": ["average", "mode", "last"]},
            # Test-time windowed metrics parameters
            "test_window_size": {"values": [1, 5, 10, 15, 30]},
            "test_stride": {"values": [1, 3, 5, 10]},
            "n_neighbors": {"values": [3, 5, 7, 10, 15, 30]},
            "feature_randomizer": {"values": [1]},
            "modality": {"values": [
                "pose", "facial", "audio", "text", "cosine",
                "pose_facial", "pose_audio", "pose_text", "pose_cosine",
                "facial_audio", "facial_text", "facial_cosine",
                "audio_text", "audio_cosine",
                "pose_facial_audio", "pose_facial_text", "pose_audio_text",
                "facial_audio_text", "facial_audio_cosine", "pose_facial_audio_cosine",
                "pose_facial_audio_text", "pose_audio_cosine",
            ]},
            "model_type": {"values": ["knn"]},
        },
    }

    print(sweep_config)

    sweep_id = wandb.sweep(sweep=sweep_config, project="brirl_linear_inter_2")
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()
