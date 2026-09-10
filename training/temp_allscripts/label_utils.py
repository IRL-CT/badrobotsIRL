"""Drop the post-resolution "recovery" block from the dataset.

Why
---
Every session runs: survey -> Error I -> Error II -> Error III -> the robot
finally acknowledges. The first and last blocks are BOTH labelled 0 ("no
error"), but they are behaviourally nothing alike: the first is a person quietly
filling in a survey, the last is a person who has just spent ~20 s getting
progressively more annoyed and is still talking and animated.

Measured on the per-second Gemini embeddings, cross-user (leave-one-participant-
out, logistic regression):

    survey-0  vs  post-resolution-0   (both labelled 0)  AUC 0.875
    error     vs  non-error           (the actual task)  AUC 0.680

The two halves of class 0 are more separable from each other than the real task
is. The trailing block is 23.3% of all frames, and removing it moves binary
cross-user AUC from 0.692 to 0.736. So it is not noise -- it is a second,
contradictory definition of "no error" baked into the labels.

What this module does
---------------------
For each participant, drops the FINAL contiguous run of label 0, but only when
errors occurred earlier in that session (so a session that never errored would
be left alone). Verified on the current dataset: 24/24 participants have such a
block, 22,420 of 96,253 frames dropped, and no participant still ends on 0.

Alignment
---------
The embedding / text / cosine CSVs are row-aligned with
`interpolated/allparticipants_100fps.csv`, so a single positional mask filters
all of them consistently. Apply `filter_recovery()` to the base frame AND to
every auxiliary frame before `iloc[last_positions]` is used, or the two will
silently desynchronise.

Note on class balance: dropping the block shifts binary from ~50/50 to 34/66
(25,384 zeros vs 48,449 ones). SMOTE in the training scripts still balances the
training folds, but the test-set prior changes, so accuracy is no longer
comparable to the v3 numbers -- compare F1/AUC instead.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
BASE_CSV = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

_MASK_CACHE = {}


def recovery_keep_mask(base_csv=BASE_CSV):
    """Boolean mask over the 100 fps rows: True = keep, False = recovery block.

    Cached, so the repeated calls inside a sweep cost one CSV read per process.
    """
    if base_csv in _MASK_CACHE:
        return _MASK_CACHE[base_csv]

    info = pd.read_csv(base_csv, usecols=[0, 1, 2, 3])
    labels = info.iloc[:, 3].to_numpy()
    participants = info.iloc[:, 1].to_numpy()

    keep = np.ones(len(info), dtype=bool)
    for p in pd.unique(participants):
        idx = np.flatnonzero(participants == p)
        lab = labels[idx]
        bounds = [0] + list(np.flatnonzero(np.diff(lab) != 0) + 1) + [len(lab)]
        start, stop = bounds[-2], bounds[-1]
        # final run is "recovery" only if it is no-error AND an error preceded it
        if lab[start] == 0 and (lab[:start] > 0).any():
            keep[idx[start:stop]] = False

    _MASK_CACHE[base_csv] = keep
    return keep


def filter_recovery(df, mask=None):
    """Drop the recovery rows from a frame that is row-aligned with the base CSV.

    Returns a positionally re-indexed copy so that `last_positions` computed on
    the filtered base frame stay valid for the filtered auxiliary frames.
    """
    if mask is None:
        mask = recovery_keep_mask()
    if len(df) == len(mask):
        return df.iloc[mask].reset_index(drop=True)

    # catch22 / tsfresh / curated_v4 come from curated_features_dataset_v4.csv,
    # which is 28,127 rows -- a different temporal grid, not the 96,253-row base.
    # Fall back to filtering on the (frame, participant) keys that survive.
    # A key-based fallback was tried here and is WRONG: curated_features_dataset_v4
    # (28,127 rows) is re-indexed onto its own grid, frames 1-1585 against the
    # base's 1-5495, so its (frame, participant) keys do not correspond. Matching
    # on them silently kept 100% of rows, i.e. catch22/tsfresh/curated_v4 would
    # have KEPT the recovery block while every other feature set dropped it.
    # Those sets are excluded from the sweeps instead.
    raise ValueError(
        f"filter_recovery: frame has {len(df)} rows, mask covers {len(mask)}. "
        f"This frame is not on the {os.path.basename(BASE_CSV)} grid and cannot "
        f"be recovery-filtered (catch22/tsfresh/curated_v4 are excluded for this "
        f"reason).")


def describe(base_csv=BASE_CSV):
    """One-line summary, for logging at the top of a run."""
    mask = recovery_keep_mask(base_csv)
    return (f"drop_recovery: keeping {mask.sum():,}/{len(mask):,} frames "
            f"({100 * (~mask).mean():.1f}% dropped as post-resolution)")


if __name__ == "__main__":
    print(describe())
