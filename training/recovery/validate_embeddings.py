#!/usr/bin/env python3
"""Sanity-check Gemini embedding CSVs before they are used for training.

The previous "clean" regeneration shipped a file in which 90.6% of rows carried a
single constant per-participant vector: extraction produced real embeddings for
the first few seconds of each session and then repeated the last one to the end.
Because the label changes during those repeated seconds, the model was handed
identical inputs with four different correct answers. It was invisible for a full
sweep cycle and cost ~1200 runs.

Every check here would have caught it in seconds.

Usage
-----
    python3 validate_embeddings.py                       # all files in data/embeddings
    python3 validate_embeddings.py path/to/file.csv ...  # specific files
    python3 validate_embeddings.py --strict              # exit 1 on any failure

Checks
------
  1. Row count and the four info columns match the base 100 fps dataset exactly,
     so `df.iloc[last_positions, 4:]` indexing is valid.
  2. Unique vectors per participant is close to the expected one-per-second.
  3. No participant has a long constant tail.
  4. No unique vector spans more than one label (the fatal case).
  5. No all-zero, NaN or duplicated-across-participants rows.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
BASE_CSV = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")

BASE_FPS = 100
INFO_COLS = 4
# One embedding per second means ~100 identical rows in a row. Allow slack for
# PCA files and for sessions that do not divide evenly.
MIN_UNIQUE_RATIO = 0.80      # unique vectors vs ceil(n_frames / BASE_FPS)
MAX_TAIL_FRACTION = 0.25     # longest constant run as a fraction of the session
# A vector straddling a label boundary legitimately covers two labels, so some
# impurity is expected. Fail only when it dominates the file.
MAX_IMPURE_ROW_FRACTION = 0.25
IMPURE_VECTOR_TOLERANCE = 2.0  # x the number of real label transitions
# Fingerprint on a subset of columns: reading 3072 float columns for 96k rows
# costs several GB, and a prefix is enough to tell distinct vectors apart.
FINGERPRINT_COLS = 32


def _load_fingerprint(path, n_cols=FINGERPRINT_COLS):
    """Read info columns plus the first `n_cols` feature columns."""
    header = pd.read_csv(path, nrows=0)
    total = len(header.columns)
    use = list(range(min(INFO_COLS + n_cols, total)))
    df = pd.read_csv(path, usecols=use)
    return df.iloc[:, :INFO_COLS], df.iloc[:, INFO_COLS:].to_numpy(dtype="float32")


def _runs(matrix):
    """Start indices of each run of consecutive identical rows."""
    if len(matrix) <= 1:
        return np.array([0])
    changed = np.any(np.diff(matrix, axis=0) != 0, axis=1)
    return np.concatenate([[0], np.flatnonzero(changed) + 1])


def validate(path, base_info):
    name = os.path.basename(path)
    failures, warnings = [], []

    info, X = _load_fingerprint(path)

    # --- 1. alignment with the base dataset -------------------------------
    if base_info is not None:
        if len(info) != len(base_info):
            failures.append(
                f"row count {len(info)} != base dataset {len(base_info)} "
                "-- last_positions indexing would be misaligned")
        else:
            renamed = info.copy()
            renamed.columns = base_info.columns
            if not renamed.reset_index(drop=True).equals(base_info.reset_index(drop=True)):
                failures.append("info columns (frame/participant/labels) differ from base dataset")

    if np.isnan(X).any():
        failures.append(f"{int(np.isnan(X).any(axis=1).sum())} rows contain NaN")
    zero_rows = int((np.abs(X).sum(axis=1) == 0).sum())
    if zero_rows:
        failures.append(f"{zero_rows} all-zero rows")

    labels = info.iloc[:, 3].to_numpy()
    participants = info.iloc[:, 1].to_numpy()

    # --- 2/3. per-participant resolution and constant tails ---------------
    rows = []
    for p in pd.unique(participants):
        mask = participants == p
        Xp, n = X[mask], int(mask.sum())
        starts = _runs(Xp)
        run_lengths = np.diff(np.append(starts, n))
        n_unique = len(np.unique(Xp, axis=0))
        expected = int(np.ceil(n / BASE_FPS))
        tail = int(run_lengths.max())
        rows.append((p, n, n_unique, expected, n_unique / expected, tail, tail / n))

    summary = pd.DataFrame(
        rows, columns=["participant", "frames", "unique", "expected",
                       "ratio", "longest_run", "tail_frac"])

    bad_ratio = summary[summary.ratio < MIN_UNIQUE_RATIO]
    if len(bad_ratio):
        failures.append(
            f"{len(bad_ratio)}/{len(summary)} participants have far fewer unique vectors "
            f"than one per second (worst: {bad_ratio.participant.iloc[bad_ratio.ratio.argmin()]} "
            f"at {bad_ratio.ratio.min():.1%} of expected)")

    bad_tail = summary[summary.tail_frac > MAX_TAIL_FRACTION]
    if len(bad_tail):
        failures.append(
            f"{len(bad_tail)}/{len(summary)} participants have a constant run covering "
            f">{MAX_TAIL_FRACTION:.0%} of the session (worst: {bad_tail.tail_frac.max():.1%}) "
            "-- this is the forward-fill signature")

    # --- 4. vectors spanning multiple labels (the fatal case) -------------
    # Some impurity is unavoidable and benign: a one-second embedding that
    # straddles a label boundary genuinely covers two labels. The expected floor
    # is the number of label transitions in the file (98 here: 23 participants
    # x 4 transitions plus p29's 6). What is fatal is impurity covering a large
    # share of rows -- the broken release had only 24 impure vectors, but those
    # covered 90.6% of all rows.
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    impure, impure_rows = 0, 0
    for gid in range(inverse.max() + 1):
        sel = inverse == gid
        if len(np.unique(labels[sel])) > 1:
            impure += 1
            impure_rows += int(sel.sum())

    n_transitions = sum(
        int((np.diff(labels[participants == p]) != 0).sum())
        for p in pd.unique(participants)
    )
    impure_frac = impure_rows / len(X)

    if impure_frac > MAX_IMPURE_ROW_FRACTION:
        failures.append(
            f"{impure} unique vectors map to more than one label, covering "
            f"{impure_rows}/{len(X)} rows ({impure_frac:.1%}) -- identical inputs "
            f"with different correct answers. Boundary-straddling seconds explain "
            f"at most ~{MAX_IMPURE_ROW_FRACTION:.0%}")
    elif impure > IMPURE_VECTOR_TOLERANCE * max(n_transitions, 1):
        failures.append(
            f"{impure} unique vectors span multiple labels, far above the "
            f"{n_transitions} label transitions that can legitimately produce one")
    elif impure:
        warnings.append(
            f"{impure} vectors span a label boundary ({impure_frac:.1%} of rows); "
            f"{n_transitions} transitions exist, so this is the expected boundary "
            "effect, not a defect")

    # --- 5. vectors shared across participants ----------------------------
    shared = 0
    for gid in range(inverse.max() + 1):
        if len(np.unique(participants[inverse == gid])) > 1:
            shared += 1
    if shared:
        warnings.append(f"{shared} unique vectors appear in more than one participant")

    total_unique = len(np.unique(X, axis=0))
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"  rows {len(X):,}   unique vectors {total_unique:,}   "
          f"expected ~{int(np.ceil(len(X)/BASE_FPS)):,} (one per second)")
    print(f"  unique per participant: min {summary.unique.min()}  "
          f"median {int(summary.unique.median())}  max {summary.unique.max()}")
    print(f"  longest constant run:   max {summary.tail_frac.max():.1%} of a session")

    if failures:
        print("\n  FAILED:")
        for f in failures:
            print(f"    - {f}")
        worst = summary.nsmallest(5, "ratio")
        print("\n  worst participants:")
        print(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for w in warnings:
        print(f"\n  WARNING: {w}")
    if not failures:
        print("\n  PASSED")

    return len(failures) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="CSVs to check (default: data/embeddings/*.csv)")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any file fails")
    args = parser.parse_args()

    files = args.files
    if not files:
        files = sorted(
            os.path.join(EMB_DIR, f) for f in os.listdir(EMB_DIR)
            if f.endswith(".csv") and "gemini" in f
        )
    if not files:
        print(f"No gemini CSVs found in {EMB_DIR}")
        return 1

    base_info = None
    if os.path.exists(BASE_CSV):
        base_info = pd.read_csv(BASE_CSV, usecols=[0, 1, 2, 3])
    else:
        print(f"WARNING: base dataset not found at {BASE_CSV}; skipping alignment check")

    results = {os.path.basename(p): validate(p, base_info) for p in files}

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} file(s) failed. Do not train on these.")
        return 1 if args.strict else 0
    print("\nAll files passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
