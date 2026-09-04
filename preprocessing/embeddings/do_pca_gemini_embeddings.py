"""
Perform PCA dimensionality reduction on Gemini embeddings.

Supports two modes:
- visual_only: Fits PCA on data/embeddings/gemini_checkpoints_visual_only
- audiovisual: Fits PCA on data/embeddings/gemini_checkpoints_audiovisual

Fits PCA on unique 1-second embedding vectors to retain 90% variance (or fixed n_components),
then transforms the full dataset frame-by-frame while preserving metadata and row parity.
"""
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}

def main():
    parser = argparse.ArgumentParser(description="Perform PCA on Gemini embeddings.")
    parser.add_argument(
        "--mode",
        choices=["visual_only", "audiovisual"],
        default="visual_only",
        help="Embedding mode: visual_only or audiovisual"
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=0.90,
        help="Variance ratio to retain (default: 0.90)"
    )
    args = parser.parse_args()

    mode = args.mode
    output_file = os.path.join(EMBEDDINGS_DIR, f"gemini_video_embeddings_pca_{mode}.csv")
    model_file = os.path.join(EMBEDDINGS_DIR, f"pca_model_gemini_{mode}.pkl")

    if mode == "visual_only":
        checkpoint_dir = os.path.join(EMBEDDINGS_DIR, "gemini_checkpoints_visual_only")
        pattern = "{p}_gemini_visual.csv"
    else:
        checkpoint_dir = os.path.join(EMBEDDINGS_DIR, "gemini_checkpoints_audiovisual")
        pattern = "{p}_gemini_audiovisual.csv"

    print(f"=== PCA on Gemini Embeddings [Mode: {mode}] ===")
    df_base = pd.read_csv(BASE_DATASET, usecols=["participant"])
    ordered_participants = []
    for p in df_base["participant"]:
        if p not in EXCLUDED_PARTICIPANTS and p not in ordered_participants:
            ordered_participants.append(p)

    print(f"1. Extracting unique 1-second embedding vectors across {len(ordered_participants)} participants...")
    unique_vectors = []
    for p in ordered_participants:
        p_file = os.path.join(checkpoint_dir, pattern.format(p=p))
        if not os.path.exists(p_file):
            p_file = os.path.join(checkpoint_dir, f"{p}_gemini.csv")
        df_p = pd.read_csv(p_file)
        # Fast sampling of 1-second fingerprints at 100fps intervals
        feats = df_p.iloc[::100, 4:].values
        unique_vectors.append(feats)

    all_unique_feats = np.vstack(unique_vectors)
    print(f"   Total unique embedding vectors: {all_unique_feats.shape[0]} x {all_unique_feats.shape[1]}")

    print(f"2. Fitting PCA to retain {args.variance*100:.0f}% variance...")
    pca = PCA(n_components=args.variance)
    pca.fit(all_unique_feats)

    print(f"   -> Total variance retained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"   -> Components needed: {pca.n_components_}")

    print("3. Transforming dataset participant-by-participant and saving...")
    pca_cols = [f"gemini_PC{i}" for i in range(1, pca.n_components_ + 1)]
    meta_cols = ["frame", "participant", "binary_label", "multiclass_label"]

    total_rows = 0
    for idx, p in enumerate(ordered_participants):
        p_file = os.path.join(checkpoint_dir, pattern.format(p=p))
        if not os.path.exists(p_file):
            p_file = os.path.join(checkpoint_dir, f"{p}_gemini.csv")
        df_p = pd.read_csv(p_file)

        df_meta = df_p[meta_cols].copy()
        feats = df_p.iloc[:, 4:].values
        df_pca = pd.DataFrame(pca.transform(feats).astype(np.float32), columns=pca_cols)
        df_out = pd.concat([df_meta, df_pca], axis=1)

        if idx == 0:
            df_out.to_csv(output_file, index=False, mode="w")
        else:
            df_out.to_csv(output_file, index=False, header=False, mode="a")
        total_rows += len(df_out)
        print(f"   [{idx+1:02d}/{len(ordered_participants)}] Transformed {p} ({len(df_out):,} rows)")

    print(f"\n4. Saving PCA model to {model_file}...")
    joblib.dump(pca, model_file)
    print(f"Done! PCA dataset written to {output_file} ({total_rows:,} rows).")

if __name__ == "__main__":
    main()
