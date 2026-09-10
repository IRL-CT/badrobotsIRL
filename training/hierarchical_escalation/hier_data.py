"""Session tensors at 1 Hz for the hierarchical escalation model (v6).

Why 1 Hz and not 100 fps
------------------------
The dataset is stored at 100 rows per second, but the Gemini embeddings carry
one vector per SECOND -- all 100 frames inside a second are byte-identical
(verified: 964 unique vectors over 96,253 rows). A 30-frame window at 100 fps is
0.30 s, so a level-1 RNN over frames would see 30 copies of the same vector and
learn nothing. Everything here is therefore resampled to 1 Hz first: conventional
features are averaged within each second, the embedding is taken as-is.

That leaves ~40 s per participant (min 33, max 55) and 24 participants -- the
real size of this dataset. Any model built here has to respect that.

Two-level structure
-------------------
    level 1   a window of `window_sec` consecutive seconds  ->  window embedding
    level 2   the ordered sequence of windows in a session  ->  per-window state

Level 2 steps over fixed time windows across the whole session (survey ->
Error I -> II -> III), so it never needs to know where the block boundaries are.
That keeps it deployable: a robot has the elapsed session, not the annotation.

Labels are v4 (recovery block dropped) via label_utils, so any change against the
v4 sweep is attributable to the model rather than to relabelling.

Folds
-----
`participant_folds` reproduces the participant assignment in
common/create_data_splits.py exactly (same seed, same shuffle, same rotation), so
v6 numbers are comparable to v4 fold for fold.
"""

import os
import random
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
sys.path.insert(0, SCRIPT_DIR)

from label_utils import filter_recovery  # noqa: E402
from feature_sets import load_feature_set, modality_blocks  # noqa: E402

BASE_CSV = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_FPS = 100
INFO_COLS = 4

# Column ranges in allparticipants_100fps.csv, matching the linear scripts.
POSE_SLICE = slice(4, 28)
FACIAL_SLICE_A = slice(28, 63)
AUDIO_SLICE = slice(63, 88)
FACIAL_SLICE_B = slice(88, None)

DIM_KEY = {128: "128d", 256: "256d", 768: "768d", 3072: "full"}


def _float32_dtypes(path, n_feature_cols=None):
    cols = pd.read_csv(path, nrows=0).columns
    feats = cols[INFO_COLS:] if n_feature_cols is None else cols[INFO_COLS:INFO_COLS + n_feature_cols]
    return {c: "float32" for c in feats}


def gemini_path(source, dims):
    return os.path.join(EMB_DIR, f"gemini_video_embeddings_{source}_{DIM_KEY[int(dims)]}.csv")


def load_frames(modality, gemini_source="visual_audio", gemini_dims=768,
                feature_set="full", binary_multiclass="binary"):
    """Frame-level feature matrix for a (feature_set, modality) pair, recovery dropped.

    `feature_set` selects the conventional-feature source: "full" (92 columns,
    sliced into pose / facial / audio) or one of the curated sets
    (curated_v5 / rf / selectkbest), each of which is one undivided block that
    naming any of pose / facial / audio includes. All of them sit on the base
    100 fps grid, so text / cosine / gemini still align positionally.

    Returns (info, X), both with the recovery block removed.
    """
    parts = modality.split("_")
    base = filter_recovery(load_feature_set(feature_set, binary_multiclass))
    info = base.iloc[:, :INFO_COLS].reset_index(drop=True)

    blocks = [b.to_numpy("float32")
              for _, b in modality_blocks(base, feature_set, modality)]
    if "text" in parts:
        p = os.path.join(DATA_DIR, "clip_text_embeddings_pca.csv")
        blocks.append(filter_recovery(pd.read_csv(p)).iloc[:, INFO_COLS:].to_numpy("float32"))
    if "cosine" in parts:
        p = os.path.join(DATA_DIR, "clip_text_cosine_similarity.csv")
        blocks.append(filter_recovery(pd.read_csv(p)).iloc[:, INFO_COLS:].to_numpy("float32"))
    if "gemini" in parts:
        p = gemini_path(gemini_source, gemini_dims)
        g = filter_recovery(pd.read_csv(p, dtype=_float32_dtypes(p)))
        blocks.append(g.iloc[:, INFO_COLS:].to_numpy("float32"))

    if not blocks:
        raise ValueError(f"modality {modality!r} selected no features")
    return info, np.hstack(blocks)


def to_seconds(info, X):
    """Collapse 100 fps to 1 Hz: mean features per second, label of the last frame.

    Seconds are cut on the frame grid within each participant, so a second that
    straddles a label boundary takes the label of its final frame -- the same
    convention `apply_windowing` uses in the flat scripts.
    """
    participants = info.iloc[:, 1].to_numpy()
    labels_b = info.iloc[:, 2].to_numpy()
    labels_m = info.iloc[:, 3].to_numpy()

    sec_X, sec_b, sec_m, sec_p = [], [], [], []
    for p in pd.unique(participants):
        idx = np.flatnonzero(participants == p)
        n_sec = len(idx) // BASE_FPS
        if n_sec == 0:
            continue
        cut = idx[:n_sec * BASE_FPS].reshape(n_sec, BASE_FPS)
        sec_X.append(X[cut].mean(axis=1))
        sec_b.append(labels_b[cut[:, -1]])
        sec_m.append(labels_m[cut[:, -1]])
        sec_p.append(np.full(n_sec, p))
    return (np.vstack(sec_X).astype("float32"),
            np.concatenate(sec_b), np.concatenate(sec_m), np.concatenate(sec_p))


def build_sessions(Xs, y, parts, window_sec, stride_sec):
    """Per participant, a sequence of windows of `window_sec` seconds.

    Returns a list of dicts, one per participant:
        X      (n_windows, window_sec, n_features)
        y      (n_windows,)   label of the window's last second
        widx   (n_windows,)   window position in the session, 0-based
        part   participant id
    """
    sessions = []
    for p in pd.unique(parts):
        m = parts == p
        Xp, yp = Xs[m], y[m]
        n = len(Xp)
        starts = list(range(0, n - window_sec + 1, stride_sec))
        if not starts:
            continue
        wX = np.stack([Xp[s:s + window_sec] for s in starts]).astype("float32")
        wy = np.array([yp[s + window_sec - 1] for s in starts])
        sessions.append({"X": wX, "y": wy, "widx": np.arange(len(starts)), "part": p})
    return sessions


def participant_folds(participants, num_folds=5, seed_value=42):
    """Reproduce the fold assignment in common/create_data_splits.py exactly.

    Same seeding, same shuffle, same rotation and 60/20/20 sizing, so v6 folds
    line up with v4 and the numbers are directly comparable.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    fold_sessions = pd.unique(participants).copy()
    n = len(fold_sessions)
    train_size = int(np.floor(0.6 * n))
    val_size = int(np.ceil(0.2 * n))
    np.random.shuffle(fold_sessions)

    folds = []
    for i in range(num_folds):
        st = i * val_size
        en = st + train_size if st + train_size <= n else st + train_size - n
        train = (np.concatenate((fold_sessions[st:], fold_sessions[:en]))
                 if st >= en else fold_sessions[st:en])
        vs = en
        ve = vs + val_size if vs + val_size <= n else vs + val_size - n
        val = (np.concatenate((fold_sessions[vs:], fold_sessions[:ve]))
               if vs >= ve else fold_sessions[vs:ve])
        test = np.setdiff1d(fold_sessions, np.concatenate((train, val)))
        folds.append((train, val, test))
    return folds


def pad_sessions(sessions, n_classes):
    """Pad a list of sessions to equal window count.

    Returns X (S, W, T, F), Y (S, W, C) one-hot, SW (S, W) sample weights that
    are 0 on padding so it contributes nothing to the loss or the metrics.
    """
    S = len(sessions)
    W = max(len(s["y"]) for s in sessions)
    T, F = sessions[0]["X"].shape[1:]
    X = np.zeros((S, W, T, F), dtype="float32")
    Y = np.zeros((S, W, n_classes), dtype="float32")
    SW = np.zeros((S, W), dtype="float32")
    for i, s in enumerate(sessions):
        w = len(s["y"])
        X[i, :w] = s["X"]
        Y[i, np.arange(w), s["y"].astype(int)] = 1.0
        SW[i, :w] = 1.0
    return X, Y, SW
