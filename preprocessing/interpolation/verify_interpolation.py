"""Verify that opensmile_interpolated.csv values are correct."""
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

interp_df = pd.read_csv('opensmile_interpolated.csv')
os_df = pd.read_csv('opensmile_processed.csv')
ap_df = pd.read_csv('all_participants_0_3.csv')
ap_df.columns = ap_df.columns.str.strip()

LABEL_COLS = ['binary_label', 'multiclass_label']
META_COLS = ['frame', 'participant'] + LABEL_COLS
NON_AUDIO_COLS = [c for c in ap_df.columns if c not in META_COLS + AUDIO_COLS]

participant = 'p2nodbot'
print(f"=== Verifying {participant} ===\n")

ap_part = ap_df[ap_df['participant'] == participant].reset_index(drop=True)
os_part = os_df[os_df['participant'] == participant].reset_index(drop=True)
interp_part = interp_df[interp_df['participant'] == participant].reset_index(drop=True)

n_video = len(ap_part)
n_audio = len(interp_part)
ratio = n_audio / n_video
print(f"Video frames: {n_video}, Audio frames: {n_audio}, Ratio: {ratio:.4f}\n")

# CHECK 1: Audio features should match opensmile_processed.csv exactly
print("CHECK 1: Audio features match opensmile_processed.csv?")
audio_match = (interp_part[AUDIO_COLS].values == os_part[AUDIO_COLS].values).all()
print(f"  All audio features match: {audio_match}\n")

# CHECK 2: At video frame boundaries, interpolated non-audio values should match original
print("CHECK 2: Interpolated values at video frame boundaries")
print(f"  audio_positions = np.arange({n_audio}) / {ratio}")
print(f"  Video frame 0 → audio frame 0, Video frame 1 → audio frame ~{ratio:.2f}, etc.\n")

# Audio frame 0 corresponds to video frame 0
# Audio frame round(ratio) corresponds to video frame 1
# etc.
for vid_frame in [0, 1, 5, 10, 50, 100]:
    audio_idx = int(round(vid_frame * ratio))
    if audio_idx >= n_audio:
        break
    audio_pos = audio_idx / ratio  # fractional video position

    print(f"  Video frame {vid_frame} → audio idx {audio_idx} (pos={audio_pos:.3f}):")
    for col in ['nose_x_delta', 'AU01_r', 'gaze_0_x']:
        orig_val = ap_part.loc[vid_frame, col]
        interp_val = interp_part.loc[audio_idx, col]
        # np.interp at position audio_idx/ratio
        expected = np.interp(audio_pos,
                             np.arange(n_video).astype(float),
                             ap_part[col].values.astype(float))
        print(f"    {col}: original={orig_val:.6f}, interpolated={interp_val:.6f}, expected={expected:.6f}, match={np.isclose(interp_val, expected)}")

# CHECK 3: Mid-frame interpolation (between video frames)
print(f"\nCHECK 3: Mid-frame interpolation between video frames 5 and 6")
v5_audio = int(round(5 * ratio))
v6_audio = int(round(6 * ratio))
mid_audio = (v5_audio + v6_audio) // 2
mid_pos = mid_audio / ratio

for col in ['nose_x_delta', 'AU06_r', 'gaze_angle_y']:
    v5 = ap_part.loc[5, col]
    v6 = ap_part.loc[6, col]
    interp_val = interp_part.loc[mid_audio, col]
    expected = np.interp(mid_pos,
                         np.arange(n_video).astype(float),
                         ap_part[col].values.astype(float))
    print(f"  {col}: v5={v5:.4f}, v6={v6:.4f}, midpoint interp={interp_val:.6f}, expected={expected:.6f}, match={np.isclose(interp_val, expected)}")

# CHECK 4: Labels forward-filled correctly
print(f"\nCHECK 4: Labels")
for vid_frame in [0, 1, 50, 100]:
    audio_start = int(round(vid_frame * ratio))
    audio_end = min(int(round((vid_frame + 1) * ratio)), n_audio)
    orig_bin = ap_part.loc[vid_frame, 'binary_label']
    orig_multi = ap_part.loc[vid_frame, 'multiclass_label']
    for ai in range(audio_start, min(audio_end, audio_start + 2)):
        i_bin = interp_part.loc[ai, 'binary_label']
        i_multi = interp_part.loc[ai, 'multiclass_label']
        print(f"  video {vid_frame} → audio {ai}: orig=({orig_bin},{orig_multi}), interp=({i_bin},{i_multi}), match={orig_bin==i_bin and orig_multi==i_multi}")
