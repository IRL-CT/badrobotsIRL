"""
Combine per-participant visual-only Gemini checkpoint CSVs into merged datasets.

MRL normalization note
----------------------
Gemini Embedding 2 uses Matryoshka Representation Learning (MRL).  When you
request a lower-dimensional embedding directly from the API it returns the
first d dimensions of the full vector **re-normalised to unit length**.
Simply slicing without re-normalising yields different vector magnitudes than
what the API would return, which breaks cosine-similarity assumptions.

This script always applies L2-normalisation after truncation so the sliced
sub-vectors are equivalent to directly-requested lower-dim API embeddings.
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "embeddings", "gemini_checkpoints_visual_only_clean")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET   = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}

META_COLS = ["frame", "participant", "binary_label", "multiclass_label"]
N_META    = len(META_COLS)


def l2_normalise(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation.  Zero vectors are left as-is."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid divide-by-zero
    return arr / norms


def get_ordered_participants():
    df_base = pd.read_csv(BASE_DATASET, usecols=["participant"])
    ordered = []
    for p in df_base["participant"].unique():
        if p not in EXCLUDED_PARTICIPANTS and p not in ordered:
            ordered.append(p)
    return sorted(ordered)


def main():
    ordered_participants = get_ordered_participants()
    print(f"Active participants ({len(ordered_participants)}): {ordered_participants}")

    # ------------------------------------------------------------------
    # 1. Stream full 3072-D dataset (no normalisation needed — already
    #    unit-norm at 3072-D as returned by the API)
    # ------------------------------------------------------------------
    full_output = os.path.join(EMBEDDINGS_DIR, "gemini_video_embeddings_visual_only_full.csv")
    print(f"\n1. Streaming full 3072-D combined file to {full_output}...")

    total_lines = 0
    with open(full_output, "w", encoding="utf-8") as out_f:
        for idx, p in enumerate(ordered_participants):
            p_file = os.path.join(CHECKPOINT_DIR, f"{p}_gemini_visual.csv")
            if not os.path.exists(p_file):
                raise FileNotFoundError(f"Missing checkpoint file: {p_file}")
            with open(p_file, "r", encoding="utf-8") as in_f:
                header = in_f.readline()
                if idx == 0:
                    out_f.write(header)
                for line in in_f:
                    out_f.write(line)
                    total_lines += 1
            print(f"  -> Merged {p}")

    print(f"Full 3072-D merged! Total data rows: {total_lines}")

    # ------------------------------------------------------------------
    # 2. MRL slices: 768-D, 256-D, 128-D
    #    Each slice takes the first d embedding dimensions then L2-
    #    normalises so the vectors match what the API returns at that dim.
    # ------------------------------------------------------------------
    for d in [768, 256, 128]:
        slice_path = os.path.join(
            EMBEDDINGS_DIR, f"gemini_video_embeddings_visual_only_{d}d.csv"
        )
        print(f"\n2. Generating {d}-D MRL slice (with L2 re-normalisation) -> {slice_path}...")

        for idx, p in enumerate(ordered_participants):
            p_file = os.path.join(CHECKPOINT_DIR, f"{p}_gemini_visual.csv")

            # Read only the columns we need (4 meta + first d embedding dims)
            use_cols = list(range(N_META + d))
            df_part = pd.read_csv(p_file, usecols=use_cols)

            # L2-normalise the embedding block to match API output at dim d
            emb = df_part.iloc[:, N_META:].values.astype(np.float32)
            emb = l2_normalise(emb)
            df_part.iloc[:, N_META:] = emb

            if idx == 0:
                df_part.to_csv(slice_path, index=False, mode="w")
            else:
                df_part.to_csv(slice_path, index=False, header=False, mode="a")
            print(f"  -> [{d}-D] Appended {p} ({len(df_part)} rows)")

    # ------------------------------------------------------------------
    # 3. PCA
    # ------------------------------------------------------------------
    print("\n3. Running PCA reduction retaining 90% variance...")
    pca_script = os.path.join(BASE_DIR, "preprocessing", "do_pca_gemini_visual_only_embeddings.py")
    if os.path.exists(pca_script):
        subprocess.run([sys.executable, pca_script], check=True)
    else:
        print(f"  WARNING: PCA script not found at {pca_script}. Skipping PCA step.")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE: ALL VISUAL-ONLY GEMINI EMBEDDINGS (3072D, 768D, 256D, 128D, PCA) GENERATED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
