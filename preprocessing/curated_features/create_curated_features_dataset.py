import numpy as np
import pandas as pd


df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

curated_features_df = pd.DataFrame()

# original columns to retain
'''
Loudness_sma3                       overall energy of speech
F0semitoneFrom27.5Hz_sma3nz         pitch of speech
alphaRatio_sma3nz                   ratio of high to low frequencies
hammarbergIndex_sma3nz              spectral balance of speech
spectralFlux_sma3nz                 rate of change in the power spectrum
mfcc1_sma3nz                        first MFCC coefficient
mfcc2_sma3nz                        second MFCC coefficient
mfcc3_sma3nz                        third MFCC coefficient
F2frequency_sma3nz                  second formant frequency, most emotionally relevant
F2bandwidth_sma3nz                  second formant bandwidth
'''

curated_features_df = df[['frame', 'participant', 'binary_label', 'multiclass_label', 
'Loudness_sma3', 'F0semitoneFrom27.5Hz_sma3nz', 'alphaRatio_sma3', 'hammarbergIndex_sma3', 'spectralFlux_sma3',
'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'F2frequency_sma3nz', 'F2bandwidth_sma3nz']].copy()


# average gaze

curated_features_df['gaze_x'] = (df[' gaze_0_x'] + df[' gaze_1_x']) / 2
curated_features_df['gaze_y'] = (df[' gaze_0_y'] + df[' gaze_1_y']) / 2
curated_features_df['gaze_z'] = (df[' gaze_0_z'] + df[' gaze_1_z']) / 2


# gaze magnitude

curated_features_df['gaze_angle_mag'] = np.sqrt(df[' gaze_angle_x']**2 + df[' gaze_angle_y']**2)


# gaze shift

curated_features_df['gaze_shift'] = np.nan

for participant, g in curated_features_df.groupby('participant'):
    g = g.sort_values('frame')
    idx = g.index
    curated_features_df.loc[idx, 'gaze_shift'] = np.sqrt(
        g['gaze_x'].diff()**2 +
        g['gaze_y'].diff()**2 +
        g['gaze_z'].diff()**2
    ) 
# Handle NaN values in gaze_shift - replace with 0 (first frame of each participant)
curated_features_df['gaze_shift'] = curated_features_df['gaze_shift'].fillna(0)


# movement energy

head_kps = ['nose', 'neck', 'righteye', 'lefteye', 'rightear', 'leftear']

curated_features_df['head_movement_energy'] = 0.0

for kp in head_kps:
    dx = df[f'{kp}_x_delta']
    dy = df[f'{kp}_y_delta']
    curated_features_df['head_movement_energy'] += dx**2 + dy**2


# verbal response time

curated_features_df['VAD_binary'] = 0
curated_features_df['verbal_response_time'] = np.nan

for participant, g in curated_features_df.groupby('participant'):
    threshold = (
        g['Loudness_sma3'].mean() +
        1.5 * g['Loudness_sma3'].std()
    )

    curated_features_df.loc[g.index, 'VAD_binary'] = (
        g['Loudness_sma3'] > threshold
    ).astype(int)

for participant, g in curated_features_df.groupby('participant'):
    g = g.sort_values('frame')
    orig_idx = g.index
    g = g.reset_index(drop=True)

    labels = g['multiclass_label']
    vad = g['VAD_binary']

    # find error onsets - look for label transitions
    label_changes = labels != labels.shift(1)
    error_onsets = g.index[label_changes & (g.index > 0)]  # Skip first frame

    # compute verbal response time for each error
    for error_frame_idx in error_onsets:
        error_label = labels.loc[error_frame_idx]
        actual_frame = g.loc[error_frame_idx, 'frame']

        # frames belonging to this error segment
        error_segment = g.index[
            (labels == error_label) & (g.index >= error_frame_idx)
        ]

        # find first frame with voice activity after error onset
        vad_after_error = vad.loc[error_frame_idx:]  # Look from error frame onwards
        speech_frames = vad_after_error[vad_after_error == 1]

        if len(speech_frames) == 0:
            continue  # participant never verbally responded

        verbal_frame_idx = speech_frames.index[0]

        # response time in frames (using actual frame numbers, not indices)
        verbal_response_time = g.loc[verbal_frame_idx, 'frame'] - actual_frame

        # assign the same response time to all frames belonging to this error
        curated_features_df.loc[g.iloc[error_segment].index, 'verbal_response_time'] = verbal_response_time

curated_features_df['has_vrt'] = curated_features_df['verbal_response_time'].notna().astype(int)

# Handle NaN values in verbal_response_time - replace with 0
curated_features_df['verbal_response_time'] = curated_features_df['verbal_response_time'].fillna(0)

# Handle NaN values in verbal_response_time - replace with 0
curated_features_df['verbal_response_time'] = curated_features_df['verbal_response_time'].fillna(0)

# cosine distance

df_cosine_dist = pd.read_csv("preprocessing/curated_features/clip_text_cosine_similarity.csv")

curated_features_df = pd.merge(
    curated_features_df,
    df_cosine_dist,
    on=['frame', 'participant'],
    how='inner')

# finalize the curated features dataset

curated_features_df = curated_features_df[
    [
        'frame', 'participant',
        'binary_label', 'multiclass_label',

        'Loudness_sma3', 'F0semitoneFrom27.5Hz_sma3nz', 
        'alphaRatio_sma3', 'hammarbergIndex_sma3', 'spectralFlux_sma3',
        'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'F2frequency_sma3nz', 'F2bandwidth_sma3nz',

        'gaze_x', 'gaze_y', 'gaze_z',
        'gaze_angle_mag', 'gaze_shift',
        'head_movement_energy',
        'VAD_binary', 'verbal_response_time', 'has_vrt',
        'Distance'
    ]
]


print(curated_features_df)
#curated_features_df.to_csv("curated_features_dataset_v3.csv", index=False)