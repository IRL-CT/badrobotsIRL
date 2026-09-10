#!/usr/bin/env python3
"""Write per-column z-scored copies of the Gemini embedding CSVs.

Raw Gemini values are tiny (std ~0.023, row L2 norm ~0.26). Concatenated with
pose / facial / audio features, which are orders of magnitude larger, the
embedding block contributes almost nothing to a distance- or gradient-based
model. Z-scoring each column to mean 0 / std 1 puts the blocks on equal footing.

Output is written alongside the input as `<name>_norm.csv`, with the four info
columns (frame, participant, binary_label, multiclass_label) copied through
unchanged so the files stay row-aligned with the base 100 fps dataset.

Runs in two streaming passes so the 3072-D file (3.6 GB) never has to fit in
memory: pass one accumulates per-column sums and sums-of-squares, pass two
transforms and writes chunk by chunk.

Note on scope: statistics are computed over the whole file, matching the
existing `create_normalized_df` in the training scripts. That technically lets
test-fold statistics inform the scaling. It is standard practice here and
consistent with the current pipeline, but it is a global fit, not a per-fold one.

Usage
-----
    python3 normalize_embeddings.py                 # every gemini CSV in data/embeddings
    python3 normalize_embeddings.py FILE [FILE ...] # specific files
    python3 normalize_embeddings.py --chunksize N   # default 20000 rows
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
EMB_DIR = os.path.join(DATA_DIR, "embeddings")

INFO_COLS = 4
DEFAULT_CHUNKSIZE = 20000
# 8 significant digits keeps z-scored values well inside float32 precision and
# never merges two distinct seconds into the same vector.
FLOAT_FORMAT = "%.8g"
EPS = 1e-12  # guard for constant columns


def compute_stats(path, chunksize):
    """Streaming pass 1: per-column mean and population std."""
    total = None
    total_sq = None
    n = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        X = chunk.iloc[:, INFO_COLS:].to_numpy("float64")
        if total is None:
            total = X.sum(axis=0)
            total_sq = (X ** 2).sum(axis=0)
        else:
            total += X.sum(axis=0)
            total_sq += (X ** 2).sum(axis=0)
        n += len(X)

    mean = total / n
    var = np.maximum(total_sq / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    n_constant = int((std < EPS).sum())
    std = np.where(std < EPS, 1.0, std)  # leave constant columns at 0 after centring
    return mean, std, n, n_constant


def write_normalized(path, out_path, mean, std, chunksize):
    """Streaming pass 2: transform and write."""
    first = True
    written = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        info = chunk.iloc[:, :INFO_COLS]
        X = chunk.iloc[:, INFO_COLS:].to_numpy("float64")
        Z = (X - mean) / std
        out = pd.concat(
            [info.reset_index(drop=True),
             pd.DataFrame(Z, columns=chunk.columns[INFO_COLS:])],
            axis=1,
        )
        out.to_csv(out_path, mode="w" if first else "a", header=first,
                   index=False, float_format=FLOAT_FORMAT)
        first = False
        written += len(out)
    return written


def verify(out_path, chunksize):
    """Confirm the result really is mean 0 / std 1, streaming."""
    total = None
    total_sq = None
    n = 0
    for chunk in pd.read_csv(out_path, chunksize=chunksize):
        X = chunk.iloc[:, INFO_COLS:].to_numpy("float64")
        if total is None:
            total = X.sum(axis=0)
            total_sq = (X ** 2).sum(axis=0)
        else:
            total += X.sum(axis=0)
            total_sq += (X ** 2).sum(axis=0)
        n += len(X)
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
    return float(np.abs(mean).max()), float(std.min()), float(std.max())


def normalize(path, chunksize):
    name = os.path.basename(path)
    if name.endswith("_norm.csv"):
        print(f"  skip {name} (already normalized)")
        return None
    out_path = path[:-len(".csv")] + "_norm.csv"

    print(f"\n{'='*78}\n{name}\n{'='*78}")
    mean, std, n, n_constant = compute_stats(path, chunksize)
    print(f"  rows {n:,}   dims {len(mean):,}")
    print(f"  before: col mean [{mean.min():+.5f}, {mean.max():+.5f}]   "
          f"col std [{std.min():.5f}, {std.max():.5f}]")
    if n_constant:
        print(f"  {n_constant} constant column(s) left at 0 (std guard)")

    written = write_normalized(path, out_path, mean, std, chunksize)
    max_mean, min_std, max_std = verify(out_path, chunksize)
    print(f"  after : |mean| max {max_mean:.2e}   std [{min_std:.6f}, {max_std:.6f}]")
    print(f"  wrote {written:,} rows -> {os.path.basename(out_path)} "
          f"({os.path.getsize(out_path)/1e9:.2f} GB)")

    # Tolerances reflect float64 accumulation over ~1e5 rows plus the
    # 8-significant-digit CSV round-trip, not the quality of the transform.
    ok = written == n and max_mean < 1e-6 and abs(max_std - 1.0) < 1e-5
    print(f"  {'OK' if ok else 'CHECK FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = parser.parse_args()

    files = args.files or sorted(
        os.path.join(EMB_DIR, f) for f in os.listdir(EMB_DIR)
        if f.endswith(".csv") and "gemini" in f and not f.endswith("_norm.csv")
    )
    if not files:
        print(f"No gemini CSVs found in {EMB_DIR}")
        return 1

    results = {}
    for path in files:
        results[os.path.basename(path)] = normalize(path, args.chunksize)

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for name, ok in results.items():
        if ok is None:
            continue
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    return 0 if all(v for v in results.values() if v is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
