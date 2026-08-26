import os
import sys
import glob
import subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "embeddings", "gemini_checkpoints_visual_audio_clean")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

# Excluded participants in the study
EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}

def main():
    print("Loading base dataset participant order...")
    df_base = pd.read_csv(BASE_DATASET, usecols=["participant"])
    ordered_participants = []
    for p in df_base["participant"].unique():
        if p not in EXCLUDED_PARTICIPANTS and p not in ordered_participants:
            ordered_participants.append(p)
    ordered_participants = sorted(ordered_participants)
    print(f"Active participants order ({len(ordered_participants)}): {ordered_participants}")

    # 1. Combine Full 3072-D Dataset via streaming
    full_output = os.path.join(EMBEDDINGS_DIR, "gemini_video_embeddings_visual_audio_full.csv")
    print(f"\n1. Streaming full 3072-D combined file to {full_output}...")
    
    total_lines = 0
    with open(full_output, "w", encoding="utf-8") as out_f:
        for idx, p in enumerate(ordered_participants):
            p_file = os.path.join(CHECKPOINT_DIR, f"{p}_gemini.csv")
            if not os.path.exists(p_file):
                raise FileNotFoundError(f"Missing checkpoint file: {p_file}")
            
            with open(p_file, "r", encoding="utf-8") as in_f:
                header = in_f.readline()
                if idx == 0:
                    out_f.write(header)
                for line in in_f:
                    out_f.write(line)
                    total_lines += 1
            print(f"  -> Merged {p} ({p_file})")

    print(f"Full 3072-D merged! Total data rows: {total_lines}")

    # 2. Slices: 768d, 256d, 128d (Streaming participant by participant)
    for d in [768, 256, 128]:
        slice_path = os.path.join(EMBEDDINGS_DIR, f"gemini_video_embeddings_visual_audio_{d}d.csv")
        print(f"\n2. Generating {d}-D slice at {slice_path}...")
        use_cols = list(range(4 + d))
        
        for idx, p in enumerate(ordered_participants):
            p_file = os.path.join(CHECKPOINT_DIR, f"{p}_gemini.csv")
            df_part = pd.read_csv(p_file, usecols=use_cols)
            
            if idx == 0:
                df_part.to_csv(slice_path, index=False, mode="w")
            else:
                df_part.to_csv(slice_path, index=False, header=False, mode="a")
            print(f"  -> [{d}-D] Appended {p} ({len(df_part)} rows)")

    # 3. Run PCA
    print("\n3. Running PCA reduction retaining 90% variance...")
    pca_script = os.path.join(BASE_DIR, "preprocessing", "do_pca_gemini_embeddings.py")
    subprocess.run([sys.executable, pca_script], check=True)

    print("\n" + "="*70)
    print("STEP 2 COMPLETE: ALL GEMINI EMBEDDINGS (3072D, 768D, 256D, 128D, PCA) SUCCESSFULLY GENERATED!")
    print("="*70)

if __name__ == "__main__":
    main()
