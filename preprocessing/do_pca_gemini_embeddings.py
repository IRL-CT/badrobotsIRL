# SCRIPT to do PCA on Gemini Video Embeddings csv cleanly & efficiently
import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "embeddings", "gemini_checkpoints_visual_audio_clean")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}

def main():
    output_file = os.path.join(EMBEDDINGS_DIR, "gemini_video_embeddings_pca_visual_audio.csv")
    model_file = os.path.join(EMBEDDINGS_DIR, "pca_model_gemini_visual_audio.pkl")

    df_base = pd.read_csv(BASE_DATASET, usecols=["participant"])
    ordered_participants = []
    for p in df_base["participant"].unique():
        if p not in EXCLUDED_PARTICIPANTS and p not in ordered_participants:
            ordered_participants.append(p)
    ordered_participants = sorted(ordered_participants)

    print(f"1. Extracting unique 1-second embedding vectors across {len(ordered_participants)} participants...")
    unique_vectors = []
    for p in ordered_participants:
        p_file = os.path.join(CHECKPOINT_DIR, f"{p}_gemini.csv")
        df_p = pd.read_csv(p_file)
        # Fast sampling of 1-second fingerprints at 100fps intervals
        feats = df_p.iloc[::100, 4:].values
        unique_vectors.append(feats)
    
    all_unique_feats = np.vstack(unique_vectors)
    print(f"Total unique embedding vectors: {all_unique_feats.shape[0]} x {all_unique_feats.shape[1]}")

    print("2. Fitting PCA to retain 90% variance...")
    pca = PCA(n_components=0.90)
    pca.fit(all_unique_feats)

    print(f"  -> Total variance retained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  -> Number of components to reach 90% variance: {pca.n_components_}")

    print("3. Transforming dataset participant-by-participant and saving...")
    pca_cols = [f"gemini_PC{i}" for i in range(1, pca.n_components_ + 1)]
    meta_cols = ["frame", "participant", "binary_label", "multiclass_label"]
    
    for idx, p in enumerate(ordered_participants):
        p_file = os.path.join(CHECKPOINT_DIR, f"{p}_gemini.csv")
        df_p = pd.read_csv(p_file)
        
        df_meta = df_p[meta_cols].copy()
        feats = df_p.iloc[:, 4:].values
        df_pca = pd.DataFrame(pca.transform(feats).astype(np.float32), columns=pca_cols)
        
        df_out = pd.concat([df_meta, df_pca], axis=1)
        
        if idx == 0:
            df_out.to_csv(output_file, index=False, mode="w")
        else:
            df_out.to_csv(output_file, index=False, header=False, mode="a")
        print(f"  -> Transformed {p} ({len(df_out)} rows)")

    print(f"\n4. Saving PCA model to {model_file}...")
    joblib.dump(pca, model_file)
    print("Done! PCA generation successfully completed.")

if __name__ == "__main__":
    main()
