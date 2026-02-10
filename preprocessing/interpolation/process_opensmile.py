"""
Create opensmile_processed.csv by zeroing out audio features in opensmile.csv
where the participant is not speaking (based on all_participants_0_3.csv).

Mapping logic:
- all_participants_0_3.csv is at ~30 FPS (video frame rate)
- opensmile.csv is at ~100 FPS (10ms windows)
- For each participant, the ratio is ~3.33x
- Each video frame i maps to opensmile frames in the range:
    [round((i-1) * ratio) + 1, round(i * ratio)]
- If the video frame has all-zero audio features (not speaking),
  zero out the corresponding opensmile frames.
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

ap_df = pd.read_csv('all_participants_0_3.csv')
os_df = pd.read_csv('opensmile.csv')

# Exclude participants
EXCLUDED = ['p3nodbot', 'p24nodbot']
os_df = os_df[~os_df['participant'].isin(EXCLUDED)].reset_index(drop=True)

# non-speaking frames in all_participants (all audio features == 0)
ap_df['is_silent'] = (ap_df[AUDIO_COLS] == 0).all(axis=1)

zeroed_count = 0
total_count = 0

for participant in sorted(os_df['participant'].unique()):

    ap_part = ap_df[ap_df['participant'] == participant].reset_index(drop=True)
    os_mask = os_df['participant'] == participant
    os_part_len = os_mask.sum()

    if len(ap_part) == 0:
        print(f"  {participant}: not in all_participants_0_3.csv, skipping (keeping as-is)")
        continue

    n_video = len(ap_part)
    n_audio = os_part_len
    ratio = n_audio / n_video

    os_indices = os_df.index[os_mask].values

    # map to the corresponding opensmile frame range
    for i in range(n_video):
        if ap_part.loc[i, 'is_silent']:
            # map video frame i (0-indexed) to opensmile frame range
            start_os = int(round(i * ratio))
            end_os = int(round((i + 1) * ratio))
            # clamp to valid range
            start_os = max(0, min(start_os, n_audio))
            end_os = max(0, min(end_os, n_audio))

            # get the actual dataframe indices
            indices_to_zero = os_indices[start_os:end_os]
            os_df.loc[indices_to_zero, AUDIO_COLS] = 0
            zeroed_count += len(indices_to_zero)

    total_count += os_part_len
    silent_video = ap_part['is_silent'].sum()
    print(f"  {participant}: {n_video} video frames, {n_audio} audio frames, "
          f"ratio={ratio:.3f}, {silent_video}/{n_video} silent video frames")

os_df.to_csv('opensmile_processed.csv', index=False)

print(f"\nZeroed out {zeroed_count} / {total_count} opensmile rows ({zeroed_count/total_count*100:.1f}%)")
print(f"Saved to opensmile_processed.csv")
print(f"Shape: {os_df.shape}")
