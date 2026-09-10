"""Curated feature sets, aligned to the 100 fps base grid so they compose.

Why this exists
---------------
The original scripts put curated_v5 / rf / selectkbest into
NO_MODALITY_SELECTION_SETS, which meant modality selection was skipped entirely
for them: they could never be combined with text, cosine or Gemini embeddings.
So "curated features + embeddings" -- arguably the most interesting fusion in the
whole study -- was never tested. (`selectkbest` even had a name-based modality
split written for it that could never run, because the set was in the skip list.)

Two things had to be fixed to make them composable:

1. ALIGNMENT. Auxiliary frames are indexed positionally via `last_positions`, so
   every feature set must sit on the same 96,253-row grid as
   `interpolated/allparticipants_100fps.csv`. Measured:

       rf_allparticipants_100fps.csv        96,253 rows, identical to base
       select_k_best_*_binary_label.csv     96,253 rows, identical to base
       select_k_best_*_multiclass_label.csv 96,253 rows, identical to base
       curated_features_dataset_v5.csv      96,229 rows -- missing exactly the
                                            LAST frame of each of the 24
                                            participants (a delta/diff artefact)

   `load_feature_set` reindexes onto the base grid by (frame, participant) and
   forward-fills within participant, so curated_v5 comes back at 96,253 rows.
   Forward-filling one trailing frame per participant is negligible, and in the
   v4 label scheme those frames sit inside the dropped recovery block anyway.

2. MODALITY SEMANTICS. pose/facial/audio column slices are only meaningful for
   the 92-feature 'full' set. For the curated sets the whole feature block is one
   group: naming any of pose/facial/audio includes it once, and text / cosine /
   gemini are appended on top as usual.

catch22 / tsfresh / curated_v4 stay non-composable: they are computed by
re-windowing inside the training scripts and produce no `last_positions`, so
there is nothing to align an embedding against.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
BASE_CSV = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")
FS_DIR = os.path.join(DATA_DIR, "feature_sets")

# Feature sets that sit on the base grid and therefore compose with embeddings.
COMPOSABLE = {"full", "curated_v5", "rf", "selectkbest"}
# Feature sets that re-window internally and cannot be aligned to aux frames.
NO_MODALITY = {"catch22", "tsfresh", "curated_v4"}

_BASE_KEY = None


def _base_key():
    global _BASE_KEY
    if _BASE_KEY is None:
        _BASE_KEY = pd.read_csv(BASE_CSV, usecols=[0, 1, 2, 3])
    return _BASE_KEY


def feature_set_path(name, binary_multiclass="binary"):
    if name == "full":
        return BASE_CSV
    if name in ("curated_v4", "catch22", "tsfresh"):
        return os.path.join(FS_DIR, "curated_features_dataset_v4.csv")
    if name == "curated_v5":
        return os.path.join(FS_DIR, "curated_features_dataset_v5.csv")
    if name == "rf":
        return os.path.join(FS_DIR, "rf_allparticipants_100fps.csv")
    if name == "selectkbest":
        suffix = "binary" if binary_multiclass == "binary" else "multiclass"
        return os.path.join(FS_DIR, f"select_k_best_allparticipants_100fps_{suffix}_label.csv")
    raise ValueError(f"unknown feature set {name!r}")


def load_feature_set(name, binary_multiclass="binary"):
    """Load a feature set, reindexed onto the base 100 fps grid when composable.

    Non-composable sets (catch22 / tsfresh / curated_v4) are returned as-is,
    since nothing will be aligned against them.
    """
    path = feature_set_path(name, binary_multiclass)
    df = pd.read_csv(path)
    if name in NO_MODALITY:
        return df

    base = _base_key()
    if len(df) == len(base):
        cols = df.columns[:2]
        if df[cols].reset_index(drop=True).equals(
                base[base.columns[:2]].rename(columns=dict(zip(base.columns[:2], cols)))
                .reset_index(drop=True)):
            return df  # already on the grid

    # Reindex by (frame, participant) and forward-fill within participant.
    fcol, pcol = df.columns[0], df.columns[1]
    key = base.iloc[:, :2].copy()
    key.columns = [fcol, pcol]
    merged = key.merge(df, on=[fcol, pcol], how="left")
    n_missing = int(merged.iloc[:, 4:].isna().any(axis=1).sum())
    merged = merged.groupby(pcol, sort=False, group_keys=False).apply(
        lambda g: g.ffill().bfill())
    merged = merged.loc[:, df.columns]           # restore original column order
    # labels come from the base grid, never from the forward fill
    merged[df.columns[2]] = base.iloc[:, 2].to_numpy()
    merged[df.columns[3]] = base.iloc[:, 3].to_numpy()
    if n_missing:
        print(f"[feature_sets] {name}: filled {n_missing} rows missing from "
              f"{os.path.basename(path)} to match the base grid")
    return merged.reset_index(drop=True)


def modality_blocks(df, feature_set, modality):
    """Feature blocks named by `modality`, excluding text/cosine/gemini.

    For 'full' this is the original pose / facial / audio column slicing. For a
    curated set the whole feature block is one group, included once if the
    modality names any of pose / facial / audio.

    Returns an ordered dict-like list of (name, DataFrame).
    """
    parts = modality.split("_")
    conventional = [p for p in parts if p in ("pose", "facial", "audio")]

    if feature_set == "full":
        out = []
        if "pose" in parts:
            out.append(("pose_full", df.iloc[:, 4:28]))
        if "facial" in parts:
            out.append(("facial_full",
                        pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)))
        if "audio" in parts:
            out.append(("audio_full", df.iloc[:, 63:88]))
        return out

    # curated_v5 / rf / selectkbest: one undivided block.
    if conventional:
        return [(f"{feature_set}_block", df.iloc[:, 4:])]
    return []


def validate(modality, feature_set):
    """Is this (modality, feature_set) pair runnable?

    text / cosine / gemini need positional alignment, so they are only valid
    with a composable feature set. A modality naming nothing but conventional
    channels is invalid for a set that has no such channels to slice.
    """
    parts = modality.split("_")
    needs_aux = any(p in parts for p in ("text", "cosine", "gemini"))
    if needs_aux and feature_set not in COMPOSABLE:
        return False
    if not needs_aux and feature_set in COMPOSABLE and feature_set != "full":
        # curated + pose/facial/audio all mean the same single block; allow it
        # but only once, which modality_blocks already guarantees.
        return any(p in parts for p in ("pose", "facial", "audio"))
    return True


if __name__ == "__main__":
    base = _base_key()
    for name in ("full", "curated_v5", "rf", "selectkbest"):
        df = load_feature_set(name)
        ok = (len(df) == len(base)
              and np.array_equal(df.iloc[:, 0].to_numpy(), base.iloc[:, 0].to_numpy())
              and np.array_equal(df.iloc[:, 1].to_numpy(), base.iloc[:, 1].to_numpy()))
        print(f"{name:12s} rows={len(df):6d} feats={df.shape[1] - 4:5d} on_base_grid={ok}")
