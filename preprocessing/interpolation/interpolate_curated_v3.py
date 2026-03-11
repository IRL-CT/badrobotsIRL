"""
Linearly interpolate non-audio features from curated_features_dataset_v3.csv
to match the audio frame rate from opensmile_interpolated.csv.

- Audio features: taken from opensmile_interpolated.csv (already at audio frame rate)
- Non-audio features (gaze, head_movement_energy, VAD, VRT, Distance, etc.):
  linearly interpolated from video rate to audio frame rate
- Labels (binary_label, multiclass_label): forward-filled from video frames
- frame: kept as the audio frame index
"""

import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
V3_PATH = '/home/sjl356/badrobotsIRL-2/preprocessing/curated_features/curated_features_dataset_v3.csv'
OPENSMILE_INTERP_PATH = '/home/sjl356/badrobotsIRL-2/preprocessing/interpolation/opensmile_interpolated.csv'
OUTPUT_PATH = '/home/sjl356/badrobotsIRL-2/preprocessing/interpolation/curated_features_v3_interpolated.csv'

# ── audio columns present in v3 (match with opensmile_interpolated) ───────────
AUDIO_COLS = [
    'Loudness_sma3', 'F0semitoneFrom27.5Hz_sma3nz',
    'alphaRatio_sma3', 'hammarbergIndex_sma3',
    'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3',
    'F2frequency_sma3nz', 'F2bandwidth_sma3nz',
]

LABEL_COLS = ['binary_label', 'multiclass_label']
META_COLS = ['frame', 'participant'] + LABEL_COLS

# ── load data ──────────────────────────────────────────────────────────────────
v3_df = pd.read_csv(V3_PATH)
os_df = pd.read_csv(OPENSMILE_INTERP_PATH)

v3_df.columns = v3_df.columns.str.strip()
os_df.columns = os_df.columns.str.strip()

# non-audio feature columns to interpolate
NON_AUDIO_COLS = [c for c in v3_df.columns if c not in META_COLS + AUDIO_COLS]
print(f"Non-audio features to interpolate ({len(NON_AUDIO_COLS)}):")
print(NON_AUDIO_COLS)
print()

all_results = []

for participant in sorted(v3_df['participant'].unique()):
    v3_part = v3_df[v3_df['participant'] == participant].reset_index(drop=True)
    os_part = os_df[os_df['participant'] == participant].reset_index(drop=True)

    if len(os_part) == 0:
        print(f"  {participant}: not in opensmile_interpolated.csv, skipping")
        continue

    n_video = len(v3_part)
    n_audio = len(os_part)
    ratio = n_audio / n_video

    # fractional video-frame positions for each audio frame (0-indexed)
    audio_positions = np.arange(n_audio) / ratio  # in video-frame space
    video_positions = np.arange(n_video).astype(float)

    # start building the result row with audio frame index and participant
    result_part = pd.DataFrame({
        'frame': np.arange(n_audio),
        'participant': participant,
    })

    # linearly interpolate each non-audio feature
    for col in NON_AUDIO_COLS:
        video_vals = v3_part[col].values.astype(float)
        interpolated = np.interp(audio_positions, video_positions, video_vals)
        result_part[col] = interpolated

    # forward-fill labels: each audio frame gets the label of its nearest video frame
    video_frame_indices = np.minimum(np.floor(audio_positions).astype(int), n_video - 1)
    for col in LABEL_COLS:
        result_part[col] = v3_part[col].values[video_frame_indices]

    # take audio features from opensmile_interpolated (already at audio rate)
    for col in AUDIO_COLS:
        result_part[col] = os_part[col].values

    all_results.append(result_part)
    print(f"  {participant}: {n_video} video → {n_audio} audio frames, "
          f"ratio={ratio:.3f}")

result = pd.concat(all_results, ignore_index=True)

# reorder columns: frame, participant, labels, audio features, non-audio features
col_order = ['frame', 'participant'] + LABEL_COLS + AUDIO_COLS + NON_AUDIO_COLS
result = result[col_order]

result.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {OUTPUT_PATH}")
print(f"Shape: {result.shape}")
print(f"Columns: {list(result.columns)}")
print(result.head())
