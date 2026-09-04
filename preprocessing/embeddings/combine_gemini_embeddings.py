"""
Combine per-participant Gemini checkpoint CSVs into unified datasets.

Supports two modes:
- visual_only: Combines checkpoints from data/embeddings/gemini_checkpoints_visual_only
- audiovisual: Combines checkpoints from data/embeddings/gemini_checkpoints_audiovisual

Outputs generated:
1. Full 3072-D: gemini_video_embeddings_{mode}_full.csv
2. Sliced MRL & L2-normalized:
   - gemini_video_embeddings_{mode}_768d.csv
   - gemini_video_embeddings_{mode}_256d.csv
   - gemini_video_embeddings_{mode}_128d.csv

Parity Guarantee:
Enforces strict 1-to-1 row-by-row, frame-by-frame, and participant-by-participant
alignment with data/interpolated/allparticipants_100fps.csv.
"""
import os
import sys
import glob
import argparse
import subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET   = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}

META_COLS = ["frame", "participant", "binary_label", "multiclass_label"]
N_META    = len(META_COLS)
MRL_DIMS  = [768, 256, 128]

def l2_normalise(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero vectors are left as-is."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms

def get_ordered_active_participants(df_base):
    """Preserve exact appearance order in allparticipants_100fps.csv."""
    ordered = []
    for p in df_base["participant"]:
        if p not in EXCLUDED_PARTICIPANTS and p not in ordered:
            ordered.append(p)
    return ordered

def main():
    parser = argparse.ArgumentParser(description="Combine per-participant Gemini checkpoint CSVs.")
    parser.add_argument(
        "--mode",
        choices=["visual_only", "audiovisual"],
        default="visual_only",
        help="Embedding mode: visual_only or audiovisual"
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "visual_only":
        checkpoint_dir = os.path.join(EMBEDDINGS_DIR, "gemini_checkpoints_visual_only")
        checkpoint_pattern = "{p}_gemini_visual.csv"
    else:
        checkpoint_dir = os.path.join(EMBEDDINGS_DIR, "gemini_checkpoints_audiovisual")
        checkpoint_pattern = "{p}_gemini_audiovisual.csv"

    print(f"=== Combining Gemini Embeddings [Mode: {mode}] ===")
    print(f"Reading base dataset from {BASE_DATASET}...")
    df_base = pd.read_csv(BASE_DATASET, usecols=META_COLS)
    ordered_participants = get_ordered_active_participants(df_base)
    print(f"Active participants ({len(ordered_participants)}): {ordered_participants}\n")

    # Verify all checkpoint files exist before starting
    for p in ordered_participants:
        p_file = os.path.join(checkpoint_dir, checkpoint_pattern.format(p=p))
        if not os.path.exists(p_file):
            # Fallback check for alternate naming
            alt_file = os.path.join(checkpoint_dir, f"{p}_gemini.csv")
            if not os.path.exists(alt_file):
                raise FileNotFoundError(f"Missing checkpoint file for {p}: {p_file}")

    # 1. Stream full 3072-D dataset
    full_output = os.path.join(EMBEDDINGS_DIR, f"gemini_video_embeddings_{mode}_full.csv")
    print(f"1. Streaming full 3072-D combined file to:\n   {full_output}")

    total_lines = 0
    with open(full_output, "w", encoding="utf-8") as out_f:
        for idx, p in enumerate(ordered_participants):
            p_file = os.path.join(checkpoint_dir, checkpoint_pattern.format(p=p))
            if not os.path.exists(p_file):
                p_file = os.path.join(checkpoint_dir, f"{p}_gemini.csv")

            with open(p_file, "r", encoding="utf-8") as in_f:
                header = in_f.readline()
                if idx == 0:
                    out_f.write(header)
                n_rows = 0
                for line in in_f:
                    out_f.write(line)
                    n_rows += 1
                total_lines += n_rows
                print(f"   [{idx+1:02d}/{len(ordered_participants)}] Appended {p}: {n_rows:,} rows")

    print(f"   Full dataset written: {total_lines:,} total frame rows.")

    # 2. Strict Parity Verification Against Base Dataset
    print("\n2. Verifying 100% row-for-row parity with base dataset...")
    df_base_active = df_base[~df_base["participant"].isin(EXCLUDED_PARTICIPANTS)].reset_index(drop=True)
    expected_rows = len(df_base_active)
    if total_lines != expected_rows:
        raise ValueError(f"PARITY ERROR: Total rows mismatch! Combined={total_lines:,} vs Base={expected_rows:,}")

    # Verify metadata alignment chunk-by-chunk
    print("   Checking participant, frame, and label sequence alignment...")
    chunk_size = 50000
    checked_rows = 0
    for chunk in pd.read_csv(full_output, usecols=META_COLS, chunksize=chunk_size):
        base_slice = df_base_active.iloc[checked_rows:checked_rows + len(chunk)].reset_index(drop=True)
        if not (chunk["participant"].values == base_slice["participant"].values).all():
            raise ValueError(f"PARITY ERROR: Participant mismatch starting at row {checked_rows}!")
        if not (chunk["frame"].values == base_slice["frame"].values).all():
            raise ValueError(f"PARITY ERROR: Frame index mismatch starting at row {checked_rows}!")
        if not (chunk["binary_label"].values == base_slice["binary_label"].values).all():
            raise ValueError(f"PARITY ERROR: Binary label mismatch starting at row {checked_rows}!")
        if not (chunk["multiclass_label"].values == base_slice["multiclass_label"].values).all():
            raise ValueError(f"PARITY ERROR: Multiclass label mismatch starting at row {checked_rows}!")
        checked_rows += len(chunk)

    print(f"   [PASSED] 100% PARITY CONFIRMED across all {checked_rows:,} frames and {len(ordered_participants)} participants!")

    # 3. Stream MRL Sliced and L2-Normalized Sub-Vectors
    for target_dim in MRL_DIMS:
        out_file = os.path.join(EMBEDDINGS_DIR, f"gemini_video_embeddings_{mode}_{target_dim}d.csv")
        print(f"\n3. Generating MRL {target_dim}-D sliced & L2-normalized dataset:\n   {out_file}")

        dim_feature_cols = [f"gemini_{j}" for j in range(target_dim)]
        read_cols = META_COLS + dim_feature_cols

        first_chunk = True
        total_dim_rows = 0
        for chunk in pd.read_csv(full_output, usecols=read_cols, chunksize=chunk_size):
            meta_df = chunk[META_COLS]
            emb_vals = chunk[dim_feature_cols].values.astype(np.float32)

            # Matryoshka normalisation: L2 re-normalise after truncation
            emb_norm = l2_normalise(emb_vals)
            norm_df = pd.DataFrame(emb_norm, columns=dim_feature_cols)

            chunk_out = pd.concat([meta_df.reset_index(drop=True), norm_df], axis=1)
            chunk_out.to_csv(out_file, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False
            total_dim_rows += len(chunk)

        file_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"   Done {target_dim}-D: {total_dim_rows:,} rows, {file_mb:.1f} MB.")

    print("\n" + "=" * 70)
    print(f"ALL EMBEDDING DATASETS COMBINED & VERIFIED FOR {mode.upper()}!")
    print("=" * 70)

if __name__ == "__main__":
    main()
