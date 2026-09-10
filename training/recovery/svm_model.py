import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb
from itertools import product
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
# Second-level evaluation + participant bootstrap CIs (see eval_utils.py).
# Drop-in wrapper: keeps every original frame-level key, adds sec_test_*.
from eval_utils import get_all_metrics
# OLD: from get_all_metrics import get_all_metrics
from create_data_splits import create_data_splits
from registry_utils import append_to_registry
# v5: RECOVERY is its own class instead of being deleted (see recovery_labels).
from recovery_labels import apply_scheme, apply_mask, describe as _scheme_desc
# OLD (v4): from label_utils import filter_recovery, describe as _recovery_desc


# Feature sets that use the full dataset without modality selection
# Curated sets now sit on the base 100 fps grid, so they COMPOSE with
# text/cosine/gemini instead of bypassing modality selection (see feature_sets).
from feature_sets import (load_feature_set, modality_blocks,
                          NO_MODALITY as NO_MODALITY_SELECTION_SETS,
                          validate as _fs_validate)
# OLD: NO_MODALITY_SELECTION_SETS = {"catch22","tsfresh","curated_v4","rf","selectkbest","curated_v5"}
#      -- curated_v5/rf/selectkbest were in the skip list, so "curated + embeddings"
#         could never be tested. They are composable now; only the re-windowing
#         sets (catch22/tsfresh/curated_v4) still bypass modality selection.

GEMINI_EMBEDDING_DIR = "../../data/embeddings"

# 2026-09-05 release: embeddings come in two sources -- "visual" (video only) and
# "visual_audio" (video + audio) -- and are already L2-normalized at source, so
# there are no separate *_norm.csv copies any more.
GEMINI_SOURCES = ("visual", "visual_audio")
GEMINI_DIM_KEYS = ("128d", "256d", "768d", "full")


def gemini_path(source, dim_key):
    """Path to one embedding CSV, e.g. ("visual_audio", "768d")."""
    return f"{GEMINI_EMBEDDING_DIR}/gemini_video_embeddings_{source}_{dim_key}.csv"


# OLD (v2): a flat dict keyed by "visual_audio_<dims>", plus a parallel
# GEMINI_EMBEDDING_PATHS_NORM of per-column z-scored copies selected when
# dataset was "norm"/"pca". Both are gone: the source is now a swept dimension of
# its own (gemini_source), and the release is L2-normalized already.
# GEMINI_EMBEDDING_PATHS = {
#     "visual_audio_128d": ".../gemini_video_embeddings_visual_audio_128d.csv",
#     ...
# }




def _gemini_float32(path, n_feature_cols=None):
    """dtype map reading the embedding columns as float32.

    pandas defaults to float64: 96,253 x 3072 is ~2.4 GB for the array alone,
    plus large parser intermediates. That is enough to get the job OOM-killed --
    it is what killed transformer_binary (job 695723). float32 halves the
    resident array and is comfortably inside the precision of the
    8-significant-digit values stored in the CSV.

    A blanket dtype="float32" cannot be used: the four info columns include the
    string `participant`, which makes pandas raise ValueError. Only the feature
    columns are mapped; the info columns are left to pandas.
    """
    cols = pd.read_csv(path, nrows=0).columns
    feats = cols[4:] if n_feature_cols is None else cols[4:4 + n_feature_cols]
    return {c: "float32" for c in feats}


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
    extracted = extracted.dropna(axis=1, how="any")
    extracted = extracted.reset_index(drop=True)

    info_df = pd.DataFrame(info_rows).drop(columns=["_window_id"]).reset_index(drop=True)
    result = pd.concat([info_df, extracted], axis=1)
    return result.dropna(axis=0, how="any").reset_index(drop=True)


def train():
    wandb.init()
    config = wandb.config
    print(config)

    # Skip modality validation for feature sets that ignore modality selection
    if config.feature_set not in NO_MODALITY_SELECTION_SETS:
        is_valid_combination = _fs_validate(config.modality, config.feature_set)
        if not is_valid_combination:
            print(f"Skipping invalid combination: feature_set={config.feature_set}, modality={config.modality}")
            wandb.log({"status": "skipped_invalid_combination"})
            return

    seed_value = 42

    # ------------------------------------------------------------------
    # Load base 100fps dataset
    # ------------------------------------------------------------------
    if True:
        # 'full' — all start from the full 100fps feature set
        df_base = load_feature_set(config.feature_set, config.binary_multiclass)
        # v4: drop the post-resolution "recovery" block. It is labelled 0 but is
        # behaviourally a talking/agitated state, and cross-user it is MORE
        # separable from the survey-0 block (AUC 0.875) than error is from
        # non-error (0.680). See label_utils for the measurement.
        df_base, _KEEP = apply_scheme(df_base, config.binary_multiclass)
        print(_scheme_desc(config.binary_multiclass))
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
        # pose/facial/audio slices only mean something for the 92-feature
        # 'full' set; a curated set is one undivided block.
        _blocks = modality_blocks(df, config.feature_set, config.modality)

        modality_components = config.modality.split("_")
        use_pca_embeddings = (config.dataset == "pca")

        # Text/gemini embeddings must be aligned to the windowed rows via last_positions
        if "text" in modality_components or "cosine" in modality_components:
            text_path = "../../data/clip_text_embeddings_pca.csv" if use_pca_embeddings else "../../data/clip_text_embeddings.csv"
            df_text_raw = apply_mask(pd.read_csv(text_path), _KEEP)
            df_text_cos_raw = apply_mask(pd.read_csv("../../data/clip_text_cosine_similarity.csv"), _KEEP)
            df_text_index = df_text_raw.iloc[last_positions, 4:].reset_index(drop=True)
            df_text_distance = df_text_cos_raw.iloc[last_positions, 4:].reset_index(drop=True)
        if "gemini" in modality_components:
            gemini_key = getattr(config, "gemini_embedding", "full")
            # OLD: when dataset == "pca" and gemini_embedding == "default", a pre-computed
            #      frame-level PCA of the embeddings was substituted. Disabled because
            #      (a) the "default" video-only file is not in the corrected release, and
            #      (b) the reported protocol applies PCA *after* windowing, which the
            #          `dataset == "pca"` branch (create_norm_pca_df) still does below.
            # if use_pca_embeddings and gemini_key == "default":
            #     gemini_path = "../../data/embeddings/gemini_video_embeddings_pca.csv"
            # else:
            #     gemini_path = GEMINI_EMBEDDING_PATHS.get(gemini_key, GEMINI_EMBEDDING_PATHS["default"])
            # dataset == "norm" loads the pre-normalized (per-column z-scored) copies.
            # Raw Gemini values have std ~0.007-0.017, so when concatenated with the
            # much larger pose/facial/audio features the embedding block barely
            # registers in a distance- or gradient-based model.
            gemini_source = getattr(config, "gemini_source", "visual_audio")
            _gpath = gemini_path(gemini_source, gemini_key)
            df_gemini_raw = apply_mask(pd.read_csv(_gpath, dtype=_gemini_float32(_gpath)), _KEEP)
            df_gemini_index = df_gemini_raw.iloc[last_positions, 4:].reset_index(drop=True)

        selected_modalities = dict(_blocks)
        if "text" in modality_components:
            selected_modalities["text_full"] = df_text_index
        if "cosine" in modality_components:
            selected_modalities["text_distance"] = df_text_distance
        if "gemini" in modality_components:
            selected_modalities["gemini_video"] = df_gemini_index

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

    # ------------------------------------------------------------------
    # Aggregate features (if enabled)
    # ------------------------------------------------------------------
    # v4: agg_features collapses ALL features to 4 summary stats (mean/std/min/max).
    # That is reasonable for a handful of pose/facial/audio channels, but it destroys
    # a 3072-D embedding. Force it off whenever the modality includes embeddings, and
    # log what was actually used so the sweep stays interpretable.
    _agg = bool(getattr(config, "agg_features", False))
    if _agg and "gemini" in str(config.modality).split("_"):
        _agg = False
        wandb.log({"agg_features_forced_off": True})
    if _agg:
        info_cols = df.iloc[:, :4]
        feature_data = df.iloc[:, 4:]
        n_original = feature_data.shape[1]
        agg_df = pd.DataFrame({
            "agg_mean": feature_data.mean(axis=1).values,
            "agg_std": feature_data.std(axis=1).values,
            "agg_min": feature_data.min(axis=1).values,
            "agg_max": feature_data.max(axis=1).values,
        })
        df = pd.concat([info_cols.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
        print(f"agg_features enabled: replaced {n_original} features with 4 aggregate stats (mean, std, min, max)")
        wandb.log({"agg_features_n_original": n_original})

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

        min_class_count = min(np.bincount(y_train))
        smote_k = min(5, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=smote_k)
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

        effective_test_stride = min(config.test_stride, config.test_window_size)
        wandb.log({"effective_test_stride": effective_test_stride, "effective_test_window_size": config.test_window_size})
        test_metrics = get_all_metrics(y_pred, y_test, y_pred_proba=y_pred_proba,
                                       sessions=session_test, window_size=config.test_window_size,
                                       stride=effective_test_stride, tolerance=1)
        test_metrics.pop("test_edt_per_session", None)
        # Collect every scalar metric, so the sec_test_* keys added by
        # eval_utils are averaged across folds alongside the originals.
        for key, value in test_metrics.items():
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                test_metrics_list.setdefault(key, []).append(value)
        # OLD: only keys pre-declared in test_metrics_list were averaged.
        # for key in test_metrics:
        #     if key in test_metrics_list:
        #         test_metrics_list[key].append(test_metrics[key])
        wandb.log({f"t{fold}_{k}": v for k, v in test_metrics.items()})
        print(test_metrics)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        wandb.log({f"t{fold}_confusion_matrix": cm.tolist()})

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

    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)
    print("Average Metrics Across Groups:", avg_test_metrics)


def validate_modality_feature_combination(modality, feature_set):
    modality_components = modality.split("_")

    # Text and gemini only work with 'full' feature set
    if "text" in modality_components and feature_set != "full":
        return False
    if "gemini" in modality_components and feature_set != "full":
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
    import argparse
    parser = argparse.ArgumentParser(description="Launch the wandb sweep for this model.")
    parser.add_argument("--project", default="brirl_gemini_v5_recovery_linear",
                        help="wandb project to log the sweep into (default: %(default)s).")
    parser.add_argument("--entity", default=None, help="wandb entity. Defaults to your wandb default.")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of sweep runs to execute (default: unlimited).")
    args = parser.parse_args()
    sweep_config = {
        "method": "random",
        "name": "brirl_gemini_v5_recovery_linear",
        "parameters": {
            "binary_multiclass": {"values": ["binary", "multiclass"]},
            # Gemini requires the "full" feature set (validate_modality_feature_combination),
            # so the sweep is pinned to "full" to avoid wasting runs on skipped combinations.
            # Curated sets are back and now COMPOSE with embeddings (feature_sets.py).
            # catch22/tsfresh re-window internally so they stay embedding-incompatible;
            # _fs_validate skips those pairs rather than wasting a run.
            # catch22/tsfresh/curated_v4 are built on curated_features_dataset_v4's
            # own re-indexed grid, so they cannot be recovery-filtered consistently
            # with the rest of the sweep. Excluded rather than silently mislabelled.
            "feature_set": {"values": ["full", "curated_v5", "rf", "selectkbest"]},
            # OLD: "feature_set": {"values": ["full", "rf", "catch22", "tsfresh", "curated_v4", "selectkbest", "curated_v5"]},
            # Embeddings are z-scored for training; "norm" is the only option now.
            "dataset": {"values": ["norm", "pca"]},
            # OLD: "dataset": {"values": ["reg", "norm", "pca"]},
            # Dataset-level windowing (applied to 100fps data before training)
            "window": {"values": [1, 5, 10, 15, 30]},
            "stride": {"values": [1, 3, 5, 10]},
            "aggregation": {"values": ["average", "mode", "last"]},
            # Test-time windowed metrics parameters
            "test_window_size": {"values": [1, 5, 10, 15, 30]},
            "test_stride": {"values": [1, 3, 5, 10]},
            "C": {"values": [0.1, 1, 10, 100]},
            "kernel": {"values": ["linear", "rbf", "poly"]},
            "gamma": {"values": ["scale", "auto"]},
            # Embeddings are high-dimensional; agg_features collapses ALL features to 4
            # summary stats (mean/std/min/max), destroying the representation. Disabled.
            # v4: back on, but forced False whenever the modality contains
            # embeddings -- collapsing 3072 dims to 4 summary stats destroys them.
            "agg_features": {"values": [True, False]},
            # OLD: "agg_features": {"values": [True,False]}, #temp only true for testing
            "feature_randomizer": {"values": [1]},
            # Source of the embedding: "visual" = video only, "visual_audio" = video+audio.
            "gemini_source": {"values": ["visual", "visual_audio"]},
            # Dimensionality. Each is its own released file, already L2-normalized.
            "gemini_embedding": {"values": ["128d", "256d", "768d", "full"]},
            # OLD (v2): {"visual_audio_128d", ..., "visual_audio_full"} -- source was
            #           baked into the key and only video+audio existed.
            # v4: ALL modality combinations restored (non-gemini included), since the
            # recovery-block fix changes the task for every modality, not just embeddings.
            "modality": {"values": [
                "pose", "facial", "audio", "text", "cosine", "gemini", "pose_facial", "pose_audio",
                "pose_text", "pose_cosine", "pose_gemini", "facial_audio", "facial_text",
                "facial_cosine", "facial_gemini", "audio_text", "audio_cosine", "audio_gemini",
                "text_gemini", "cosine_gemini", "pose_facial_audio", "pose_facial_text",
                "pose_facial_cosine", "pose_facial_gemini", "pose_audio_text", "pose_audio_cosine",
                "pose_audio_gemini", "facial_audio_text", "facial_audio_cosine",
                "facial_audio_gemini", "pose_text_gemini", "facial_text_gemini",
                "audio_text_gemini", "pose_facial_audio_text", "pose_facial_audio_cosine",
                "pose_facial_audio_gemini", "pose_facial_text_gemini", "facial_audio_text_gemini",
                "pose_audio_text_gemini", "pose_facial_audio_text_gemini"
            ]},
            "model_type": {"values": ["svm"]},
        },
    }

    print(sweep_config)

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, function=train, count=args.count,
                entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
