import os
import csv
import subprocess
import tempfile
import sys
import numpy as np
from scipy.io import wavfile
from collections import defaultdict

def mute_robot_audio(target_participant=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diarization_file = os.path.join(base_dir, "HRI25_LBR", "data", "speaker_diarization_all_audio.csv")
    input_video_dir = os.path.join(base_dir, "data", "raw", "mp4")
    output_video_dir = os.path.join(base_dir, "data", "raw", "mp4_human_only_audio")

    os.makedirs(output_video_dir, exist_ok=True)

    print(f"Reading diarization timestamps from: {diarization_file}")
    nodbot_intervals = defaultdict(list)
    with open(diarization_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = row["speaker"].strip().lower()
            # Mute any non-participant turn (nodbot, researcher, etc.)
            if speaker in ["nodbot", "robot", "researcher", "other"]:
                p_num = row["participant_num"].strip()
                start = float(row["start"])
                end = float(row["end"])
                nodbot_intervals[p_num].append((start, end))

    # Get list of mp4 files
    if target_participant:
        target_name = target_participant.replace(".mp4", "")
        video_files = [f"{target_name}.mp4"]
        print(f"Targeting single participant: {target_name}")
    else:
        video_files = sorted([f for f in os.listdir(input_video_dir) if f.endswith(".mp4")])
        print(f"Found {len(video_files)} video files in {input_video_dir}.\n")

    summary = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, v_file in enumerate(video_files, 1):
            p_name = os.path.splitext(v_file)[0]
            input_path = os.path.join(input_video_dir, v_file)
            output_path = os.path.join(output_video_dir, v_file)

            if not os.path.exists(input_path):
                print(f"Warning: {input_path} does not exist. Skipping.")
                continue

            intervals = nodbot_intervals.get(p_name, [])
            total_muted_sec = sum(end - start for start, end in intervals)

            print(f"[{idx}/{len(video_files)}] Processing {v_file}: {len(intervals)} non-human turns ({total_muted_sec:.2f}s) to mute...")

            temp_wav_in = os.path.join(tmp_dir, f"{p_name}_in.wav")
            temp_wav_out = os.path.join(tmp_dir, f"{p_name}_out.wav")

            # 1. Extract audio to WAV
            extract_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100",
                temp_wav_in
            ]
            res = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"  Error extracting audio from {v_file}: {res.stderr.decode('utf-8')[:200]}")
                continue

            # 2. Load WAV and mute non-participant intervals
            sr, data = wavfile.read(temp_wav_in)
            data_muted = data.copy()

            for start, end in intervals:
                s_idx = max(0, int(round(start * sr)))
                e_idx = min(len(data_muted), int(round(end * sr)))
                if s_idx < e_idx:
                    data_muted[s_idx:e_idx] = 0

            # 3. Write muted WAV
            wavfile.write(temp_wav_out, sr, data_muted)

            # 4. Recombine video (stream copy) with new muted audio
            merge_cmd = [
                "ffmpeg", "-y", "-i", input_path, "-i", temp_wav_out,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                output_path
            ]
            res = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"  Error merging audio into {v_file}: {res.stderr.decode('utf-8')[:200]}")
                continue

            # Clean up temp wav files for this video
            if os.path.exists(temp_wav_in):
                os.remove(temp_wav_in)
            if os.path.exists(temp_wav_out):
                os.remove(temp_wav_out)

            summary.append((p_name, len(intervals), total_muted_sec, os.path.getsize(output_path)))
            print(f"  -> Saved {output_path} ({os.path.getsize(output_path)/(1024*1024):.1f} MB)")

    print(f"\nDone! Updated {len(summary)} video(s).")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    mute_robot_audio(target)
