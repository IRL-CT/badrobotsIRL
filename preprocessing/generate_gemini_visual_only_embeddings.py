import os
import sys
import time
import glob
import shutil
import argparse
import tempfile
import subprocess
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_VIDEO_DIR = os.path.join(DATA_DIR, "raw", "mp4_human_only_audio")
DOWNSCALED_DIR  = os.path.join(DATA_DIR, "raw", "mp4_480p_visual_only")
CHECKPOINT_DIR  = os.path.join(DATA_DIR, "embeddings", "gemini_checkpoints_visual_only_clean")
EMBEDDINGS_DIR  = os.path.join(DATA_DIR, "embeddings")
BASE_DATASET    = os.path.join(DATA_DIR, "interpolated", "allparticipants_100fps.csv")

EXCLUDED_PARTICIPANTS = {"p1nodbot", "p3nodbot", "p13nodbot", "p15nodbot", "p24nodbot", "p30nodbot"}
MIN_CLIP_COVERAGE = 0.85

sys.stdout.reconfigure(line_buffering=True)

def get_ffmpeg_binary():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local_exe = os.path.expanduser("~/.local/bin/ffmpeg")
    if os.path.exists(local_exe):
        return local_exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

FFMPEG_BIN = get_ffmpeg_binary()
FFPROBE_BIN = os.path.join(os.path.dirname(FFMPEG_BIN), "ffprobe") if os.path.dirname(FFMPEG_BIN) else "ffprobe"
print(f"Using ffmpeg binary: {FFMPEG_BIN}")

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)

def get_video_duration_sec(path):
    """Accurately extract duration in seconds from video container."""
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ]
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)
    except Exception:
        try:
            cmd = [FFMPEG_BIN, "-i", path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in res.stderr.decode('utf-8', errors='ignore').split('\n'):
                if "Duration:" in line:
                    parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except Exception:
            pass
    return None

def downscale_video_visual_only(participant_id):
    """Downscale full-res 4K video to 480p, stripping audio entirely."""
    os.makedirs(DOWNSCALED_DIR, exist_ok=True)
    out_path = os.path.join(DOWNSCALED_DIR, f"{participant_id}.mp4")
    in_path  = os.path.join(INPUT_VIDEO_DIR, f"{participant_id}.mp4")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input video not found: {in_path}")

    src_dur = get_video_duration_sec(in_path)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        out_dur = get_video_duration_sec(out_path)
        if src_dur and out_dur and abs(out_dur - src_dur) < 2.0:
            print(f"[{participant_id}] Valid visual-only 480p video cached ({out_dur:.1f}s).")
            return participant_id, out_path
        else:
            print(f"[{participant_id}] Existing 480p video mismatch/truncated (src={src_dur}s, out={out_dur}s). Re-encoding...")
            try:
                os.remove(out_path)
            except Exception:
                pass

    print(f"[{participant_id}] Downscaling source video ({src_dur:.1f}s) to 480p (visual only, no audio)...")
    cmd = [
        FFMPEG_BIN, "-y", "-i", in_path,
        "-vf", "scale=854:480,fps=30",
        "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",          # Strip audio track -- visual embeddings only
        out_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"Downscale failed for {participant_id}:\n{res.stderr.decode('utf-8', errors='ignore')[-500:]}")

    out_dur = get_video_duration_sec(out_path)
    if src_dur and out_dur and (src_dur - out_dur) > 2.0:
        raise RuntimeError(f"Downscale truncated {participant_id}: out={out_dur:.1f}s vs src={src_dur:.1f}s")

    print(f"[{participant_id}] Downscale complete ({out_dur:.1f}s, {os.path.getsize(out_path)/(1024*1024):.1f} MB, no audio).")
    return participant_id, out_path

def embed_video_clip(client, clip_bytes, max_retries=6):
    """Embed 1-second mp4 clip (visual only) using Gemini Embedding 2."""
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
                raise ValueError("No embedding returned from API")
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Embedding API failed after {max_retries} attempts: {e}") from e
            wait = 2.0 * (attempt + 1)
            print(f"    [Retry {attempt+1}/{max_retries}] {e}. Waiting {wait:.0f}s...")
            time.sleep(wait)

def is_checkpoint_valid(checkpoint_file, expected_seconds):
    """Validate that checkpoint contains enough unique fingerprints."""
    if not os.path.exists(checkpoint_file) or os.path.getsize(checkpoint_file) < 1000:
        return False
    try:
        df = pd.read_csv(checkpoint_file, usecols=["gemini_0", "gemini_1", "gemini_2"])
        n_unique = df.drop_duplicates().shape[0]
        min_required = max(3, int(expected_seconds * MIN_CLIP_COVERAGE))
        if n_unique >= min_required:
            print(f"  -> Valid checkpoint: {n_unique} unique fingerprints (need >= {min_required}).")
            return True
        print(f"  WARNING: Checkpoint has only {n_unique} unique fingerprints (need >= {min_required}). Regenerating.")
        return False
    except Exception:
        return False

def process_participant(participant_id, client, df_base_part):
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{participant_id}_gemini_visual.csv")
    n_frames = len(df_base_part)
    expected_seconds = int(np.ceil(n_frames / 100))

    if is_checkpoint_valid(checkpoint_file, expected_seconds):
        print(f"[{participant_id}] Valid checkpoint loaded.")
        return pd.read_csv(checkpoint_file)

    # 1. Ensure visual-only 480p downscaled video is complete & valid
    _, video_480p = downscale_video_visual_only(participant_id)

    # 2. Segment into 1-second clips (audio already absent in source)
    embeddings_by_sec = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_pattern = os.path.join(tmp_dir, "clip_%04d.mp4")
        subprocess.run([
            FFMPEG_BIN, "-y", "-i", video_480p,
            "-c", "copy", "-f", "segment", "-segment_time", "1",
            "-reset_timestamps", "1",
            chunk_pattern
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        clip_files = sorted(glob.glob(os.path.join(tmp_dir, "clip_*.mp4")))
        print(f"[{participant_id}] Sliced {len(clip_files)} 1-second clips (visual only). Embedding in parallel...")

        def embed_worker(sec_idx, clip_path):
            with open(clip_path, "rb") as f:
                b = f.read()
            vec = embed_video_clip(client, b)
            return sec_idx, vec

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(embed_worker, s, p) for s, p in enumerate(clip_files)]
            done = 0
            for fut in as_completed(futures):
                sec_idx, vec = fut.result()
                embeddings_by_sec[sec_idx] = vec
                done += 1
                if done % 10 == 0 or done == len(clip_files):
                    print(f"  [{participant_id}] {done}/{len(clip_files)} clips embedded...")

    # 3. Validate coverage
    n_embedded = len(embeddings_by_sec)
    min_clips = int(expected_seconds * MIN_CLIP_COVERAGE)
    if n_embedded < min_clips:
        raise RuntimeError(
            f"[{participant_id}] Only {n_embedded}/{expected_seconds} clips succeeded (need >= {min_clips}). "
            f"NOT saving checkpoint."
        )

    # 4. Map 1-second embeddings to 100fps frame rows
    feature_matrix = np.zeros((n_frames, 3072), dtype=np.float32)
    for i in range(n_frames):
        sec_idx = i // 100
        if sec_idx in embeddings_by_sec:
            feature_matrix[i, :] = embeddings_by_sec[sec_idx]
        elif embeddings_by_sec:
            nearest = min(embeddings_by_sec.keys(), key=lambda k: abs(k - sec_idx))
            feature_matrix[i, :] = embeddings_by_sec[nearest]

    feature_cols = [f"gemini_{j}" for j in range(3072)]
    df_feats = pd.DataFrame(feature_matrix, columns=feature_cols)
    meta_cols = ["frame", "participant", "binary_label", "multiclass_label"]
    df_meta = df_base_part[meta_cols].reset_index(drop=True)
    df_full = pd.concat([df_meta, df_feats], axis=1)

    df_full.to_csv(checkpoint_file, index=False)
    print(f"[{participant_id}] Done. {n_embedded} clips embedded (visual only). Checkpoint saved ({df_full.shape[0]} rows x {df_full.shape[1]} cols).")
    return df_full

def main():
    parser = argparse.ArgumentParser(description="Generate visual-only Gemini embeddings (no audio).")
    parser.add_argument("--participants", nargs="+", default=None, help="Process specific participant IDs.")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    client = get_gemini_client()

    print(f"Loading base dataset from {BASE_DATASET}...")
    df_base = pd.read_csv(BASE_DATASET)
    all_participants = sorted(df_base["participant"].unique())
    active_participants = [p for p in all_participants if p not in EXCLUDED_PARTICIPANTS]

    if args.participants:
        requested = set(args.participants)
        active_participants = [p for p in active_participants if p in requested]

    print(f"Processing {len(active_participants)} participant(s): {active_participants}\n")

    for idx, p in enumerate(active_participants, 1):
        print(f"\n[{idx}/{len(active_participants)}] Processing {p}...")
        df_part = df_base[df_base["participant"] == p].copy()
        process_participant(p, client, df_part)

    print("\n" + "=" * 70)
    print("EMBEDDING COMPLETE! Run combine_gemini_visual_only_embeddings.py to build final dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
