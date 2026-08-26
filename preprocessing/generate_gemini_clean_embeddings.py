import os
import sys
import time
import glob
import subprocess
import tempfile
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_VIDEO_DIR = os.path.join(DATA_DIR, "raw", "mp4_human_only_audio")
DOWNSCALED_DIR = os.path.join(DATA_DIR, "raw", "mp4_480p_human_only_audio")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "embeddings", "gemini_checkpoints_visual_audio_clean")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

# Excluded participants in the study
EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)

def downscale_video(participant_id):
    os.makedirs(DOWNSCALED_DIR, exist_ok=True)
    out_path = os.path.join(DOWNSCALED_DIR, f"{participant_id}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return participant_id, out_path

    in_path = os.path.join(INPUT_VIDEO_DIR, f"{participant_id}.mp4")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input video not found: {in_path}")

    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vf", "scale=854:480:flags=fast_bilinear", "-r", "30",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "fastdecode",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return participant_id, out_path

def embed_video_clip(client, clip_bytes, max_retries=6):
    part = types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4")
    for attempt in range(max_retries):
        try:
            res = client.models.embed_content(
                model="models/gemini-embedding-2",
                contents=[part]
            )
            if hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0].values
            elif hasattr(res, "embedding") and res.embedding:
                return res.embedding.values
            else:
                raise ValueError("No embedding returned")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Embedding failed after {max_retries} attempts: {e}")
                raise e
            time.sleep(2.0 * (attempt + 1))

def process_participant(participant_id, client, df_base_part):
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{participant_id}_gemini.csv")
    if os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 1000:
        print(f"[{participant_id}] Checkpoint already exists. Loaded.")
        return pd.read_csv(checkpoint_file)

    video_480p = os.path.join(DOWNSCALED_DIR, f"{participant_id}.mp4")
    if not os.path.exists(video_480p):
        downscale_video(participant_id)

    # Get total video duration
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_480p]
    dur_str = subprocess.check_output(cmd).decode().strip()
    total_sec = int(np.ceil(float(dur_str)))

    print(f"[{participant_id}] Segmenting {total_sec} 1-second clips and embedding...")

    embeddings_by_sec = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Segment into 1s clips directly from 480p video in 0.3s
        chunk_pattern = os.path.join(tmp_dir, "clip_%04d.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_480p,
            "-c", "copy", "-f", "segment", "-segment_time", "1",
            "-reset_timestamps", "1",
            chunk_pattern
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        clip_files = sorted(glob.glob(os.path.join(tmp_dir, "clip_*.mp4")))
        
        # Parallel embedding calls
        def embed_worker(idx, path):
            with open(path, "rb") as f:
                b = f.read()
            vec = embed_video_clip(client, b)
            return idx, vec

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(embed_worker, sec, path) for sec, path in enumerate(clip_files)]
            for f in as_completed(futures):
                sec, vec = f.result()
                embeddings_by_sec[sec] = vec

    # Construct 100fps mapped dataframe
    n_frames = len(df_base_part)
    feature_matrix = np.zeros((n_frames, 3072), dtype=np.float32)

    for i in range(n_frames):
        sec_idx = min(i // 100, total_sec - 1)
        if sec_idx in embeddings_by_sec:
            feature_matrix[i, :] = embeddings_by_sec[sec_idx]
        elif (len(embeddings_by_sec) - 1) in embeddings_by_sec:
            feature_matrix[i, :] = embeddings_by_sec[len(embeddings_by_sec) - 1]

    feature_cols = [f"gemini_{j}" for j in range(3072)]
    df_feats = pd.DataFrame(feature_matrix, columns=feature_cols)

    meta_cols = ["frame", "participant", "binary_label", "multiclass_label"]
    df_meta = df_base_part[meta_cols].reset_index(drop=True)
    df_full = pd.concat([df_meta, df_feats], axis=1)

    # Save participant checkpoint
    df_full.to_csv(checkpoint_file, index=False)
    print(f"[{participant_id}] Finished! Checkpoint saved ({df_full.shape[0]} rows x {df_full.shape[1]} cols).")
    return df_full

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    client = get_gemini_client()

    print(f"Loading base dataset from {BASE_DATASET}...")
    df_base = pd.read_csv(BASE_DATASET)
    participants = sorted(df_base["participant"].unique())
    active_participants = [p for p in participants if p not in EXCLUDED_PARTICIPANTS]

    print(f"Found {len(active_participants)} active study participants to embed.")

    # Step 1: Pre-downscale videos in parallel
    print("\n--- Step 1: Pre-downscaling videos to 480p in parallel ---")
    to_downscale = [p for p in active_participants if not os.path.exists(os.path.join(DOWNSCALED_DIR, f"{p}.mp4"))]
    if to_downscale:
        print(f"Downscaling {len(to_downscale)} videos (using 4 parallel workers)...")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(downscale_video, p): p for p in to_downscale}
            for f in as_completed(futures):
                p_id, path = f.result()
                print(f"  -> Downscaled {p_id}.mp4")
    else:
        print("All 480p downscaled videos already exist.")

    # Step 2: Extract & embed each participant
    print("\n--- Step 2: Generating Gemini Multimodal Embeddings ---")
    participant_dfs = []
    for idx, p in enumerate(active_participants, 1):
        print(f"\n[{idx}/{len(active_participants)}] Processing {p}...")
        df_part = df_base[df_base["participant"] == p].copy()
        df_out = process_participant(p, client, df_part)
        if df_out is not None:
            participant_dfs.append(df_out)

    # Step 3: Combine and export
    print("\n--- Step 3: Combining and Slicing Dimensions ---")
    full_df = pd.concat(participant_dfs, axis=0).reset_index(drop=True)
    print(f"Full combined dataset shape: {full_df.shape}")

    full_output_path = os.path.join(EMBEDDINGS_DIR, "gemini_video_embeddings_visual_audio_full.csv")
    print(f"Saving full 3072-D dataset to {full_output_path}...")
    full_df.to_csv(full_output_path, index=False)

    for d in [768, 256, 128]:
        slice_path = os.path.join(EMBEDDINGS_DIR, f"gemini_video_embeddings_visual_audio_{d}d.csv")
        print(f"Saving {d}-D slice to {slice_path}...")
        meta_and_slice = full_df.iloc[:, : 4 + d]
        meta_and_slice.to_csv(slice_path, index=False)

    # Step 4: Run PCA
    print("\n--- Step 4: Computing PCA (90% variance) ---")
    pca_script = os.path.join(BASE_DIR, "preprocessing", "do_pca_gemini_embeddings.py")
    subprocess.run([sys.executable, pca_script], check=True)

    print("\n" + "="*70)
    print("STEP 2 COMPLETE: GEMINI MULTIMODAL EMBEDDINGS SUCCESSFULLY GENERATED!")
    print("="*70)

if __name__ == "__main__":
    main()
