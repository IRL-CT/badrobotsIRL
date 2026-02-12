"""
Full tsfresh dataset generation with feature selection for repeated robot failure HRI data.

Steps:
1. Windowing (no label transitions crossed)
2. tsfresh feature extraction
3. Feature selection
4. Ready for ML training
"""

import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import GroupKFold

WINDOW_SIZE = 120   # 4 seconds
STRIDE = 30         # 1 second overlap
NON_FEATURE_COLS = ["frame", "participant", "binary_label", "multiclass_label"]


def build_windows(df, window_size, stride):
    windows = []
    for participant, g in df.groupby("participant"):
        g = g.sort_values("frame").reset_index(drop=True)

        # Split into contiguous segments of constant multiclass label
        segments = []
        start_idx = 0
        for i in range(1, len(g)):
            if g.loc[i, "multiclass_label"] != g.loc[i-1, "multiclass_label"]:
                segments.append(g.iloc[start_idx:i])
                start_idx = i
        segments.append(g.iloc[start_idx:])

        # Sliding windows inside each segment
        for seg in segments:
            if len(seg) < window_size:
                continue
            for s in range(0, len(seg) - window_size + 1, stride):
                window = seg.iloc[s:s+window_size].copy()
                window["window_id"] = f"{participant}_seg{seg.index[0]}_win{s}"
                windows.append(window)
    return pd.concat(windows, ignore_index=True)


def main():
    df = pd.read_csv("all_participants_0_3.csv")
    df.columns = df.columns.str.strip()
    FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]

    df_windowed = build_windows(df, WINDOW_SIZE, STRIDE)

    # Verify no label mixing
    assert (df_windowed.groupby("window_id")["multiclass_label"].nunique() == 1).all(), \
        "Mixed labels in a window!"

    # tsfresh feature extraction
    fc_params = EfficientFCParameters()

    X = extract_features(
        df_windowed[["window_id", "frame"] + FEATURE_COLS],
        column_id="window_id",
        column_sort="frame",
        default_fc_parameters=fc_params,
        disable_progressbar=False
    )

    impute(X)

    # window level labels (take last frame's label in each window)
    y_binary = df_windowed.groupby("window_id")["binary_label"].last()
    y_multi = df_windowed.groupby("window_id")["multiclass_label"].last()

    # participant groups for CV
    groups = df_windowed.groupby("window_id")["participant"].first()

    # Align indices
    y_binary = y_binary.loc[X.index]
    y_multi = y_multi.loc[X.index]
    groups = groups.loc[X.index]

    # feature selection binary
    X_selected_binary = select_features(X, y_binary)

    # one-vs-rest multiclass selection
    X_selected_per_class = {}
    for c in [1, 2, 3]:
        y_c = (y_multi == c).astype(int)
        X_selected_per_class[c] = select_features(X, y_c)

    print("Total windows:", len(X))
    print("Features per window (raw):", X.shape[1])
    print("Features selected (binary):", X_selected_binary.shape[1])
    print("Binary label distribution:\n", y_binary.value_counts(normalize=True))
    print("Multiclass label distribution:\n", y_multi.value_counts(normalize=True))

    # leakage-safe groupk-fold split
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_selected_binary, y_binary, groups)):
        print(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

    # get start frame for each window
    start_frames = df_windowed.groupby("window_id")["frame"].first()
    start_frames = start_frames.loc[X_selected_binary.index]

    out = pd.DataFrame()
    out["frame"] = start_frames
    out["participant"] = groups
    out["binary_label"] = y_binary
    out["multiclass_label"] = y_multi

    # Add features
    out = pd.concat([out, X_selected_binary], axis=1)
    out.index.name = "window_id"

    out.to_csv("tsfresh_dataset.csv")
    print(f"\nSaved tsfresh_dataset.csv ({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
