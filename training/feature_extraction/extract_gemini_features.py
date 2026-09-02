#!/usr/bin/env python3
"""
Generate 100fps-aligned Gemini video embeddings for the badrobotsIRL dataset.

This script:
1. Loads allparticipants_100fps.csv to find the exact ordering and length of each participant's data.
2. Extracts 1-second clips from the raw MP4s.
3. Retrieves full 3072-D video embeddings from the Gemini API.
4. Broadcasts/upsamples the embeddings 100x to perfectly align with the 100fps dataframe.
5. Saves the result so that RNN training scripts can load and window them dynamically.
"""

import os
import sys
import math
import time
import shutil
import subprocess
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

# ── Configuration & Paths ──────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

BASE_CSV_PATH = DATA_DIR / "interpolated" / "allparticipants_100fps.csv"
MP4_DIR = DATA_DIR / "raw" / "mp4"
OUTPUT_DIR = DATA_DIR / "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "gemini-embedding-2-preview"

# ── API Client ──────────────────────────────────────────────────────────────

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        _client = genai.Client(api_key=api_key)
        print(f"Gemini client initialized (key: {api_key[:8]}...)")
    return _client

# ── Video Processing ───────────────────────────────────────────────────────

def extract_clip_fast(input_path, output_vid_path, output_aud_path, start_sec, end_sec):
    """Extract a fast, low-res video clip and an audio clip."""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-t", str(duration),
        "-map", "0:v:0", "-vf", "scale=-2:480,fps=30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(output_vid_path),
        "-map", "0:a:0", "-c:a", "libmp3lame", str(output_aud_path)
    ]
    subprocess.run(cmd, check=True)
    return output_vid_path, output_aud_path

def get_multimodal_embedding(video_path, audio_path, dimensions=None, retries=3):
    """Embed video and audio files using Gemini, with retries for quota/rate limits."""
    client = get_client()
    config_args = {}
    if dimensions is not None:
        config_args["output_dimensionality"] = dimensions
    config = types.EmbedContentConfig(**config_args)
    
    # Fast retry block
    delay = 2
    for attempt in range(max(1, retries)):
        try:
            contents = []
            
            # Read video
            if Path(video_path).exists():
                with open(video_path, "rb") as f:
                    contents.append(types.Part.from_bytes(data=f.read(), mime_type="video/mp4"))
                    
            # Read audio
            if Path(audio_path).exists():
                with open(audio_path, "rb") as f:
                    contents.append(types.Part.from_bytes(data=f.read(), mime_type="audio/mp3"))
            
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            return np.array(response.embeddings[0].values)
        
        except Exception as e:
            if attempt < retries - 1:
                print(f"      [Retry {attempt+1}] API Error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"      [Failed] Could not embed {video_path}: {e}")
                actual_dim = dimensions if dimensions else 3072
                return np.zeros(actual_dim)  # Return zeros on absolute failure to preserve alignment

# ── Main Extraction Pipeline ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate 100fps-aligned Gemini video embeddings.")
    parser.add_argument("--dimensions", type=int, default=None, help="Output dimensionality (MRL). Defaults to full model dimension if not specified.")
    args = parser.parse_args()
    
    dimensions = args.dimensions
    actual_dim = dimensions if dimensions else 3072
    dim_str = f"{dimensions}d" if dimensions else "full"
    
    output_csv_path = OUTPUT_DIR / f"gemini_video_embeddings_visual_audio_{dim_str}.csv"
    checkpoint_dir = OUTPUT_DIR / f"gemini_checkpoints_visual_audio_{dim_str}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=====================================================")
    print(f"Starting Gemini Sequence Embedding Pipeline (100fps)")
    print(f"Using Dimensionality: {dim_str}")
    print("=====================================================")
    
    # 1. Load Base DataFrame to ascertain exact participant lengths
    print(f"Loading reference base CSV: {BASE_CSV_PATH}")
    df_base = pd.read_csv(BASE_CSV_PATH)
    
    # We only care about the index and labels for alignment
    # Columns 0-3 are assumed to be: participant_col, frame_col, binary_label, multiclass_label
    meta_cols = df_base.columns[:4].tolist()
    print(f"Metadata columns identified: {meta_cols}")
    
    participants = df_base.iloc[:, 1].unique()  # participant col is usually index 1
    print(f"Found {len(participants)} participants in base CSV.")
    
    first_write = True
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temp dir for video chunks: {temp_dir}")
    
    try:
        # 2. Iterate Participant by Participant
        for pid in participants:
            print(f"Processing participant: {pid}")
            
            # Check for existing checkpoint
            ckpt_path = checkpoint_dir / f"{pid}_gemini.csv"
            if ckpt_path.exists():
                print(f"Checkpoint found. Skipping API generation.")
                df_pid_final = pd.read_csv(ckpt_path)
                df_pid_final.to_csv(output_csv_path, mode='a' if not first_write else 'w', header=first_write, index=False)
                first_write = False
                del df_pid_final
                import gc
                gc.collect()
                continue
                
            # Filter rows for this participant
            df_pid_base = df_base[df_base.iloc[:, 1] == pid].copy()
            total_frames = len(df_pid_base)
            total_seconds = math.ceil(total_frames / 100.0)
            print(f"  Rows: {total_frames} -> {total_seconds} seconds to embed.")
            
            # Resolve MP4 Path (handle naming conventions)
            mp4_name = f"{pid}.mp4" if not str(pid).startswith("p") else f"{pid}.mp4"
            if not mp4_name.endswith("nodbot.mp4") and "nodbot" in mp4_name:
                pass # Already has it
            elif not mp4_name.endswith("nodbot.mp4"):
                mp4_name = mp4_name.replace(".mp4", "nodbot.mp4")
            
            mp4_path = MP4_DIR / mp4_name
            if not mp4_path.exists():
                print(f"Warning: {mp4_name} not found in {MP4_DIR}. Filling with zeros.")
                # Fallback zero-filling to maintain pipeline integrity
                embeddings_array = np.zeros((total_seconds, actual_dim))
            else:
                # 3. Process video second-by-second
                embeddings_list = []
                for s in range(total_seconds):
                    if s % 10 == 0 or s == total_seconds - 1:
                        print(f"Slicing & Embedding: {s}/{total_seconds} sec...")
                        
                    vid_path = temp_dir / f"clip_{s}.mp4"
                    aud_path = temp_dir / f"clip_{s}.mp3"
                    
                    try:
                        extract_clip_fast(mp4_path, vid_path, aud_path, s, s + 1.0)
                        emb = get_multimodal_embedding(vid_path, aud_path, dimensions)
                        embeddings_list.append(emb)
                        
                        # Cleanup temp clip
                        if vid_path.exists():
                            vid_path.unlink()
                        if aud_path.exists():
                            aud_path.unlink()
                            
                    except Exception as e:
                        print(f"    Error processing second {s}: {e}. Zero-filling.")
                        embeddings_list.append(np.zeros(actual_dim))
                        
                    time.sleep(0.5)  # Rate limit pacing (max 15 RPM on free tier, slightly faster if using premium)
                
                embeddings_array = np.array(embeddings_list)
            
            # 4. Upsample (Broadcast) the 1-second embeddings to 100fps
            print(f"  Upsampling {total_seconds} embeddings to match {total_frames} frames...")
            # Create an array of indices where frame i maps to second floor(i/100)
            indices_100fps = np.arange(total_frames) // 100
            # Cap the indices safely in case of ceiling overshoot
            indices_100fps = np.clip(indices_100fps, 0, len(embeddings_array) - 1)
            
            upsampled_embeddings = embeddings_array[indices_100fps]
            
            # Format dataframe
            emb_cols = [f"gemini_{i}" for i in range(actual_dim)]
            df_emb = pd.DataFrame(upsampled_embeddings, columns=emb_cols, index=df_pid_base.index)
            
            # Concat with metadata
            df_pid_final = pd.concat([df_pid_base[meta_cols], df_emb], axis=1)
            
            # Save checkpoint
            df_pid_final.to_csv(ckpt_path, index=False)
            
            # Save iteratively
            df_pid_final.to_csv(output_csv_path, mode='a' if not first_write else 'w', header=first_write, index=False)
            first_write = False
            
            # Free memory
            del df_pid_final
            import gc
            gc.collect()
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    print(f"\n=====================================================")
    print(f"Finished processing all participants.")
    print(f"Final dataset saved iteratively to {output_csv_path}")
    print("=====================================================")

if __name__ == "__main__":
    import tempfile
    main()
