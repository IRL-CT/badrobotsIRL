"""Relabel the dataset around the RECOVERY block (v5).

Session structure
-----------------
Every participant runs one session with five contiguous blocks:

    survey  ->  Error I  ->  Error II  ->  Error III  ->  robot acknowledges
    NEUTRAL      ERROR        ERROR         ERROR          RECOVERY

The original labelling calls both the survey and the post-acknowledgement block
"no error" (label 0). Measured cross-user on the per-second Gemini embeddings
(leave-one-participant-out, logistic regression):

    survey-0  vs  recovery-0  (both labelled 0)   AUC 0.875
    error     vs  non-error   (the original task) AUC 0.680

The two halves of class 0 are more separable from each other than the task
itself is, and in a 3-way cross-user confusion 35% of recovery seconds are
called "error" while 26% of error seconds are called "recovery". v4 handled this
by deleting the recovery block; v5 instead promotes it to a class of its own.

Schemes
-------
binary      ERROR vs RECOVERY, neutral dropped.
            The hard contrast: two states that both involve a person talking and
            animated, separated only by whether the breakdown is still ongoing.
            binary_label: 0 = error, 1 = recovery.

multiclass  NEUTRAL / ERROR / RECOVERY over all frames.
            multiclass_label: 0 = neutral, 1 = error, 2 = recovery.

Both schemes rewrite the label columns in place rather than changing anything
downstream, so `create_data_splits` keeps working unmodified (it reads column 2
for binary and column 3 for multiclass).

Alignment
---------
Each scheme keeps a different subset of rows, so it returns a positional mask.
The embedding / text / cosine CSVs are row-aligned with the base 100 fps file,
so the SAME mask must be applied to them before `iloc[last_positions]` is used,
or features and labels silently desynchronise.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
BASE_CSV = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

NEUTRAL, ERROR, RECOVERY = 0, 1, 2
_KIND_CACHE = {}


def segment_kind(base_csv=BASE_CSV):
    """Per-row block identity: NEUTRAL / ERROR / RECOVERY.

    ERROR is any frame with multiclass_label > 0. RECOVERY is the final
    contiguous run of label 0 in a session, but only when an error preceded it.
    Everything else (in practice the leading survey block) is NEUTRAL.
    """
    if base_csv in _KIND_CACHE:
        return _KIND_CACHE[base_csv]

    info = pd.read_csv(base_csv, usecols=[0, 1, 2, 3])
    labels = info.iloc[:, 3].to_numpy()
    participants = info.iloc[:, 1].to_numpy()

    kind = np.where(labels > 0, ERROR, NEUTRAL)
    for p in pd.unique(participants):
        idx = np.flatnonzero(participants == p)
        lab = labels[idx]
        bounds = [0] + list(np.flatnonzero(np.diff(lab) != 0) + 1) + [len(lab)]
        start, stop = bounds[-2], bounds[-1]
        if lab[start] == 0 and (lab[:start] > 0).any():
            kind[idx[start:stop]] = RECOVERY

    _KIND_CACHE[base_csv] = kind
    return kind


def scheme_mask(scheme, base_csv=BASE_CSV):
    """Rows kept by a scheme. binary drops NEUTRAL; multiclass keeps everything."""
    kind = segment_kind(base_csv)
    if scheme == "binary":
        return kind != NEUTRAL
    return np.ones(len(kind), dtype=bool)


def apply_mask(df, mask, base_csv=BASE_CSV):
    """Filter a frame by the scheme mask, positionally or by key.

    catch22 / tsfresh / curated_v4 are built from
    curated_features_dataset_v4.csv, which is 28,127 rows -- a different temporal
    grid entirely, not the 96,253-row base. Those frames cannot take a positional
    mask, so they are filtered on the (frame, participant) keys that survive it.
    Without this the run died with "frame has 28127 rows, mask covers 96253".
    """
    if len(df) == len(mask):
        return df.iloc[mask].reset_index(drop=True)

    # A key-based fallback was tried and is WRONG: curated_features_dataset_v4 is
    # re-indexed onto its own grid (frames 1-1585 vs the base's 1-5495), so its
    # keys do not correspond and matching on them kept 100% of rows -- those runs
    # would have KEPT the recovery block while every other feature set dropped it.
    raise ValueError(
        f"apply_mask: frame has {len(df)} rows, mask covers {len(mask)}. Not on "
        f"the {os.path.basename(base_csv)} grid (catch22/tsfresh/curated_v4 are "
        f"excluded for this reason).")


def apply_scheme(df_base, scheme, base_csv=BASE_CSV):
    """Rewrite the label columns for `scheme` and drop the rows it excludes.

    Returns
    -------
    (df, mask)
        `df` has columns 2/3 rewritten and is positionally re-indexed.
        `mask` must be applied to every auxiliary frame with `apply_mask`.
    """
    if scheme not in ("binary", "multiclass"):
        raise ValueError(f"unknown scheme {scheme!r}")

    kind = segment_kind(base_csv)
    mask = scheme_mask(scheme, base_csv)
    df = apply_mask(df_base, mask)
    k = kind[mask]

    bin_col, mc_col = df.columns[2], df.columns[3]
    if scheme == "binary":
        # 0 = error, 1 = recovery. Mirror into both columns so either target
        # index in create_data_splits gives the same task.
        y = (k == RECOVERY).astype(int)
        df[bin_col] = y
        df[mc_col] = y
    else:
        # 0 = neutral, 1 = error, 2 = recovery.
        df[mc_col] = k
        df[bin_col] = (k != NEUTRAL).astype(int)

    return df, mask


def describe(scheme, base_csv=BASE_CSV):
    """One-line summary for logging at the start of a run."""
    kind = segment_kind(base_csv)
    mask = scheme_mask(scheme, base_csv)
    k = kind[mask]
    if scheme == "binary":
        return (f"v5 binary ERROR-vs-RECOVERY: {mask.sum():,}/{len(mask):,} frames "
                f"(neutral dropped); error={int((k == ERROR).sum()):,} "
                f"recovery={int((k == RECOVERY).sum()):,}")
    return (f"v5 multiclass NEUTRAL/ERROR/RECOVERY: {mask.sum():,} frames; "
            f"neutral={int((k == NEUTRAL).sum()):,} error={int((k == ERROR).sum()):,} "
            f"recovery={int((k == RECOVERY).sum()):,}")


if __name__ == "__main__":
    for s in ("binary", "multiclass"):
        print(describe(s))
