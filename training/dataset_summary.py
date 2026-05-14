"""
Print total dataset size, per-modality feature counts, per-dataset-type info,
label distributions (whole dataset + per fold), and per-fold train/val/test
sample counts for the interparticipant split.
"""
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))

# =====================================================================
# DATASET OVERVIEW
# =====================================================================
df = pd.read_csv(os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv"))

print("=" * 70)
print("BASE DATASET OVERVIEW  (allparticipants_100fps.csv)")
print("=" * 70)
print(f"Total rows (samples):    {len(df)}")
print(f"Total columns:           {df.shape[1]}")
print(f"Total features:          {df.shape[1] - 4}  (columns 4+)")
print(f"Unique participants:     {df['participant'].nunique()}")
print(f"Participants:            {sorted(df['participant'].unique())}")

print(f"\nPer-participant sample counts:")
for p in sorted(df['participant'].unique()):
    n = len(df[df['participant'] == p])
    print(f"  {p:20s}: {n:6d} samples")

# =====================================================================
# LABEL DISTRIBUTIONS (whole dataset)
# =====================================================================
print("\n" + "=" * 70)
print("LABEL DISTRIBUTIONS (whole dataset)")
print("=" * 70)

print("\n  Binary label distribution:")
for label, count in df['binary_label'].value_counts().sort_index().items():
    pct = count / len(df) * 100
    print(f"    Label {label}: {count:6d}  ({pct:.1f}%)")

print("\n  Multiclass label distribution:")
for label, count in df['multiclass_label'].value_counts().sort_index().items():
    pct = count / len(df) * 100
    print(f"    Label {label}: {count:6d}  ({pct:.1f}%)")

# =====================================================================
# MODALITY BREAKDOWN (from base 'full' dataset)
# =====================================================================
print("\n" + "=" * 70)
print("MODALITY BREAKDOWN  (feature groups in base dataset)")
print("=" * 70)

all_cols = df.columns.tolist()
info_cols = all_cols[:4]
feature_cols = all_cols[4:]

# Modality splits match gru_multiclass_model_wandb.py lines 949-951:
#   pose  = cols  4:28  (indices 0..23 of feature_cols)
#   facial= cols 28:63 + 88:  (AU_r + AU_c + gaze cols after audio)
#   audio = cols 63:88
pose_cols   = all_cols[4:28]     # 24 features
facial_cols = all_cols[28:63] + all_cols[88:]  # 35 + 8 = 43 features (note: gaze included after audio in raw index)
audio_cols  = all_cols[63:88]    # 25 features

# More intuitive sub-grouping:
gaze_cols = [c for c in feature_cols if 'gaze' in c.lower()]
au_cols   = [c for c in feature_cols if c.startswith('AU')]
audio_only = [c for c in feature_cols if c not in pose_cols and c not in gaze_cols and c not in au_cols]

print(f"\n  {'Modality':<25s}  {'# Features':>10s}  Columns (code index)")
print(f"  {'-'*25}  {'-'*10}  {'-'*30}")
print(f"  {'Pose (body keypoints)':<25s}  {len(pose_cols):>10d}  cols 4:28")
print(f"  {'Facial (AU, code)':<25s}  {len(facial_cols):>10d}  cols 28:63 + 88:")
print(f"  {'Audio (code)':<25s}  {len(audio_cols):>10d}  cols 63:88")
print(f"  {'TOTAL':<25s}  {len(feature_cols):>10d}")

print(f"\n  Detailed sub-groups:")
print(f"    Pose keypoint deltas:  {len(pose_cols):3d} features")
print(f"    Action Units (AU):     {len(au_cols):3d} features")
print(f"    Gaze:                  {len(gaze_cols):3d} features")
print(f"    Audio (OpenSMILE):     {len(audio_only):3d} features")

print(f"\n  Pose columns:   {pose_cols}")
print(f"  AU columns:     {au_cols}")
print(f"  Gaze columns:   {gaze_cols}")
print(f"  Audio columns:  {audio_only}")

# =====================================================================
# DATASET TYPES (feature sets)
# =====================================================================
print("\n" + "=" * 70)
print("DATASET TYPES (feature sets used in training)")
print("=" * 70)

datasets = {
    "full": os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv"),
    "curated_features_v5_100fps": os.path.join(DATA_DIR, "interpolated", "curated_features_v5_100fps.csv"),
    "rf": os.path.join(DATA_DIR, "feature_sets", "rf_allparticipants_100fps.csv"),
    "selectkbest (multiclass)": os.path.join(DATA_DIR, "feature_sets", "select_k_best_allparticipants_100fps_multiclass_label.csv"),
    "selectkbest (binary)": os.path.join(DATA_DIR, "feature_sets", "select_k_best_allparticipants_100fps_binary_label.csv"),
}

embeddings = {
    "text (CLIP PCA)": os.path.join(DATA_DIR, "embeddings", "clip_text_embeddings_pca.csv"),
    "cosine (CLIP)": os.path.join(DATA_DIR, "embeddings", "clip_text_cosine_similarity.csv"),
    "gemini (full 3072-D)": os.path.join(DATA_DIR, "embeddings", "gemini_video_embeddings_visual_audio_full.csv"),
}

print(f"\n  {'Dataset / Feature Set':<30s}  {'Rows':>8s}  {'Features':>8s}  {'Columns (names)'}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*40}")

for name, path in datasets.items():
    if os.path.exists(path):
        tmp = pd.read_csv(path, nrows=0)
        nrows = sum(1 for _ in open(path)) - 1
        n_feat = tmp.shape[1] - 4
        feat_names = tmp.columns[4:].tolist()
        if len(feat_names) > 6:
            feat_str = ", ".join(feat_names[:3]) + " ... " + ", ".join(feat_names[-3:])
        else:
            feat_str = ", ".join(feat_names)
        print(f"  {name:<30s}  {nrows:>8d}  {n_feat:>8d}  {feat_str}")
    else:
        print(f"  {name:<30s}  (file not found)")

print(f"\n  {'Embedding Dataset':<30s}  {'Rows':>8s}  {'Features':>8s}  {'Columns (names)'}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*40}")

for name, path in embeddings.items():
    if os.path.exists(path):
        tmp = pd.read_csv(path, nrows=0)
        nrows = sum(1 for _ in open(path)) - 1
        n_feat = tmp.shape[1] - 4
        feat_names = tmp.columns[4:].tolist()
        if len(feat_names) > 6:
            feat_str = ", ".join(feat_names[:3]) + " ... " + ", ".join(feat_names[-3:])
        else:
            feat_str = ", ".join(feat_names)
        print(f"  {name:<30s}  {nrows:>8d}  {n_feat:>8d}  {feat_str}")
    else:
        print(f"  {name:<30s}  (file not found)")

# =====================================================================
# PER-FOLD BREAKDOWN (interparticipant, 5-fold, same logic as create_data_splits.py)
# =====================================================================
excluded_participants = ["p24nodbot"]
df_filtered = df[~df['participant'].isin(excluded_participants)].reset_index(drop=True)

print("\n" + "=" * 70)
print("PER-FOLD BREAKDOWN  (interparticipant, 5-fold, seed=42)")
print("=" * 70)
print(f"After excluding {excluded_participants}:")
print(f"  Total rows: {len(df_filtered)}")
print(f"  Participants: {df_filtered['participant'].nunique()}")

seed_value = 42
num_folds = 5
np.random.seed(seed_value)

fold_sessions = df_filtered['participant'].unique()
num_of_sessions = len(fold_sessions)

train_size = int(np.floor(0.6 * num_of_sessions))
val_size   = int(np.ceil(0.2 * num_of_sessions))
test_size  = num_of_sessions - train_size - val_size

print(f"\n  Participant split: train={train_size}, val={val_size}, test={test_size} participants")

np.random.shuffle(fold_sessions)

total_across_folds = {"train": 0, "val": 0, "test": 0}
avg_binary = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
avg_multi = {"train": {0: 0, 1: 0, 2: 0, 3: 0}, "val": {0: 0, 1: 0, 2: 0, 3: 0}, "test": {0: 0, 1: 0, 2: 0, 3: 0}}

for fold in range(num_folds):
    start_train_index = fold * val_size
    end_train_index = (start_train_index + train_size
                       if start_train_index + train_size <= len(fold_sessions)
                       else start_train_index + train_size - len(fold_sessions))

    if start_train_index >= end_train_index:
        train_fold = np.concatenate((fold_sessions[start_train_index:], fold_sessions[:end_train_index]))
    else:
        train_fold = fold_sessions[start_train_index:end_train_index]

    val_train_index = end_train_index
    val_end_index = (val_train_index + val_size
                     if val_train_index + val_size <= len(fold_sessions)
                     else val_train_index + val_size - len(fold_sessions))

    if val_train_index >= val_end_index:
        val_fold = np.concatenate((fold_sessions[val_train_index:], fold_sessions[:val_end_index]))
    else:
        val_fold = fold_sessions[val_train_index:val_end_index]

    test_fold = np.setdiff1d(fold_sessions, np.concatenate((train_fold, val_fold)))

    train_mask = df_filtered['participant'].isin(train_fold)
    val_mask   = df_filtered['participant'].isin(val_fold)
    test_mask  = df_filtered['participant'].isin(test_fold)

    n_train = train_mask.sum()
    n_val   = val_mask.sum()
    n_test  = test_mask.sum()

    total_across_folds["train"] += n_train
    total_across_folds["val"]   += n_val
    total_across_folds["test"]  += n_test

    print(f"\n  ── Fold {fold} ──")
    print(f"    Train participants ({len(train_fold)}): {sorted(train_fold)}")
    print(f"    Val   participants ({len(val_fold)}):   {sorted(val_fold)}")
    print(f"    Test  participants ({len(test_fold)}):  {sorted(test_fold)}")
    print(f"    Samples:  train={n_train:6d}  val={n_val:6d}  test={n_test:6d}  total={n_train+n_val+n_test:6d}")

    # Binary label distribution per split
    print(f"    Binary label distribution:")
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        counts = df_filtered.loc[mask, 'binary_label'].value_counts().sort_index()
        for l, c in counts.items():
            avg_binary[split_name][l] += c
        dist_str = "  ".join([f"label {l}: {c:5d} ({c/mask.sum()*100:.1f}%)" for l, c in counts.items()])
        print(f"      {split_name:5s}: {dist_str}")

    # Multiclass label distribution per split
    print(f"    Multiclass label distribution:")
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        counts = df_filtered.loc[mask, 'multiclass_label'].value_counts().sort_index()
        for l, c in counts.items():
            avg_multi[split_name][l] += c
        dist_str = "  ".join([f"label {l}: {c:5d} ({c/mask.sum()*100:.1f}%)" for l, c in counts.items()])
        print(f"      {split_name:5s}: {dist_str}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total dataset:   {len(df_filtered)} samples, {df_filtered['participant'].nunique()} participants")
print(f"  Folds:           {num_folds}")
print(f"  Average per fold:")
print(f"    train: {total_across_folds['train']/num_folds:8.0f} samples")
print(f"    val:   {total_across_folds['val']/num_folds:8.0f} samples")
print(f"    test:  {total_across_folds['test']/num_folds:8.0f} samples")

print(f"\n  Average Binary label distribution per fold:")
for split in ["train", "val", "test"]:
    avg_total = total_across_folds[split] / num_folds
    dist_str = "  ".join([f"label {l}: {c/num_folds:5.0f} ({(c/num_folds)/avg_total*100:.1f}%)" for l, c in avg_binary[split].items()])
    print(f"    {split:5s}: {dist_str}")

print(f"\n  Average Multiclass label distribution per fold:")
for split in ["train", "val", "test"]:
    avg_total = total_across_folds[split] / num_folds
    dist_str = "  ".join([f"label {l}: {c/num_folds:5.0f} ({(c/num_folds)/avg_total*100:.1f}%)" for l, c in avg_multi[split].items()])
    print(f"    {split:5s}: {dist_str}")
print(f"  Total features (base): {df.shape[1] - 4}")
print(f"    Pose:    {len(pose_cols)}")
print(f"    Facial:  {len(facial_cols)} (as indexed in training code)")
print(f"    Audio:   {len(audio_cols)} (as indexed in training code)")
