"""
Linearly interpolate non-audio features from all_participants_0_3.csv
into the opensmile_processed.csv frame rate, producing a final combined dataset.

- Audio features: kept from opensmile_processed.csv (already silence-masked)
- Non-audio features (pose, AUs, gaze): linearly interpolated to match audio frame rate
- Labels (binary_label, multiclass_label): forward-filled from video frames
- audio_frame: kept as the frame index
"""

import pandas as pd
import numpy as np

AUDIO_COLS = [
    'Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3',
    'slope0-500_sma3', 'slope500-1500_sma3', 'spectralFlux_sma3',
    'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3',
    'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz',
    'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz',
    'logRelF0-H1-H2_sma3nz', 'logRelF0-H1-A3_sma3nz',
    'F1frequency_sma3nz', 'F1bandwidth_sma3nz', 'F1amplitudeLogRelF0_sma3nz',
    'F2frequency_sma3nz', 'F2bandwidth_sma3nz', 'F2amplitudeLogRelF0_sma3nz',
    'F3frequency_sma3nz', 'F3bandwidth_sma3nz', 'F3amplitudeLogRelF0_sma3nz',
]

LABEL_COLS = ['binary_label', 'multiclass_label']
META_COLS = ['frame', 'participant'] + LABEL_COLS

os_df = pd.read_csv('opensmile_processed.csv')
ap_df = pd.read_csv('all_participants_0_3.csv')

# strip whitespace from column names
ap_df.columns = ap_df.columns.str.strip()

# non-audio feature columns to interpolate
NON_AUDIO_COLS = [c for c in ap_df.columns if c not in META_COLS + AUDIO_COLS]
print(f"Non-audio features to interpolate ({len(NON_AUDIO_COLS)}):")
print(NON_AUDIO_COLS)

all_results = []

for participant in sorted(os_df['participant'].unique()):
    os_part = os_df[os_df['participant'] == participant].reset_index(drop=True)
    ap_part = ap_df[ap_df['participant'] == participant].reset_index(drop=True)

    if len(ap_part) == 0:
        print(f"  {participant}: not in all_participants_0_3.csv, skipping")
        continue

    n_video = len(ap_part)
    n_audio = len(os_part)
    ratio = n_audio / n_video

    # create fractional video-frame positions for each audio frame (0-indexed)
    # audio frame i corresponds to video position i / ratio
    audio_positions = np.arange(n_audio) / ratio  # in video-frame space

    # video frame positions (0-indexed integers)
    video_positions = np.arange(n_video).astype(float)

    # linearly interpolate each non-audio feature
    for col in NON_AUDIO_COLS:
        video_vals = ap_part[col].values.astype(float)
        interpolated = np.interp(audio_positions, video_positions, video_vals)
        os_part[col] = interpolated

    # forward-fill labels: each audio frame gets the label of its corresponding video frame
    video_frame_indices = np.minimum(np.floor(audio_positions).astype(int), n_video - 1)
    for col in LABEL_COLS:
        os_part[col] = ap_part[col].values[video_frame_indices]

    all_results.append(os_part)

    print(f"  {participant}: {n_video} video → {n_audio} audio frames, "
          f"ratio={ratio:.3f}")

result = pd.concat(all_results, ignore_index=True)

# reorder columns: audio_frame, participant, labels, non-audio features, audio features
col_order = ['audio_frame', 'participant'] + LABEL_COLS + NON_AUDIO_COLS + AUDIO_COLS
result = result[col_order]

result.to_csv('opensmile_interpolated.csv', index=False)

print(f"\nSaved opensmile_interpolated.csv")
print(f"Shape: {result.shape}")
print(f"Columns: {list(result.columns)}")
print(result.head())
