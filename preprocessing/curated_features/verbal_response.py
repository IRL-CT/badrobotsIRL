"""
can calculate VAD from
Loudness_sma3 feature extracted using openSMILE

create a threshold like mean + 1.5 times standard deviation

df['VAD_binary'] = (df['Loudness_sma3'] > df['Loudness_sma3'].mean() 
                               + 1.5*df['Loudness_sma3'].std()
                              ).astype(int)

find first frame at which error occurred (when frame’s label is different from previous frame) (eframe)
find first frame at which VAD_binary = 1 (vframe)

response time = vframe - eframe

reaction frames between the vframe to next error label would be labeled with response_time

"""
import pandas as pd
import numpy as np

df = pd.read_csv("all_participants_0_3.csv")

# Debug: Check if the required columns exist
print("Columns in dataset:", df.columns.tolist())
print("Dataset shape:", df.shape)
print("Sample of data:")
print(df[['participant', 'multiclass_label', 'Loudness_sma3']].head(10))

# compute participant-specific VAD threshold using loudness
# VAD_binary = 1 when voice is active, 0 otherwise

df['VAD_binary'] = 0

for participant, g in df.groupby('participant'):
    threshold = (
        g['Loudness_sma3'].mean() +
        1.5 * g['Loudness_sma3'].std()
    )  # create threshold for voice activity

    df.loc[g.index, 'VAD_binary'] = (
        g['Loudness_sma3'] > threshold
    ).astype(int)

# verbal response toime (single static feature per error)
# Verbal response time = number of frames between robot error onset and participant's first verbal response
# error onset = frame where multiclass_label changes from n to n+1

df['verbal_response_time'] = np.nan

for participant, g in df.groupby('participant'):
    print(f"\nProcessing participant: {participant}")
    g = g.sort_values('frame').reset_index(drop=True)  # Sort by frame, not index

    labels = g['multiclass_label']
    vad = g['VAD_binary']

    # find error onsets - look for label transitions
    # Method 1: Check for any label change (not just +1)
    label_changes = labels != labels.shift(1)
    error_onsets = g.index[label_changes & (g.index > 0)]  # Skip first frame
    
    print(f"  Found {len(error_onsets)} potential error onsets")
    if len(error_onsets) > 0:
        print(f"  Error onsets at frames: {g.loc[error_onsets, 'frame'].tolist()}")

    # compute verbal response time for each error
    for idx, error_frame_idx in enumerate(error_onsets):
        error_label = labels.loc[error_frame_idx]
        actual_frame = g.loc[error_frame_idx, 'frame']
        
        print(f"    Processing error {idx+1}: frame {actual_frame}, label {error_label}")

        # frames belonging to this error segment
        error_segment_mask = (labels == error_label)
        error_segment = g.index[error_segment_mask]

        # find first frame with voice activity after error onset
        vad_after_error = vad.loc[error_frame_idx:]  # Look from error frame onwards
        speech_frames = vad_after_error[vad_after_error == 1]

        print(f"      VAD frames after error: {len(speech_frames)} speech frames found")
        
        if len(speech_frames) == 0:
            print(f"      No verbal response found for this error")
            continue  # participant never verbally responded

        verbal_frame_idx = speech_frames.index[0]
        
        # response time in frames (using actual frame numbers, not indices)
        verbal_response_time = g.loc[verbal_frame_idx, 'frame'] - actual_frame
        
        print(f"      Verbal response at frame {g.loc[verbal_frame_idx, 'frame']}, response time: {verbal_response_time}")

        # assign the same response time to all frames belonging to this error
        df.loc[g.iloc[error_segment].index, 'verbal_response_time'] = verbal_response_time


df['has_vrt'] = df['verbal_response_time'].notna().astype(int)

# Handle NaN values in verbal_response_time
nan_count_before = df['verbal_response_time'].isna().sum()
print(f"\nNaN values before handling: {nan_count_before}")

# Replace NaN with 0 (indicating immediate response or no response)
df['verbal_response_time'] = df['verbal_response_time'].fillna(0)

nan_count_after = df['verbal_response_time'].isna().sum()
print(f"NaN values after handling: {nan_count_after}")
print(f"Default value used for NaN: 0")

# Update has_vrt based on original NaN status
# has_vrt = 1 means there was an actual verbal response, 0 means no response (was NaN)

# Debug output
print("\nFinal Results:")
print(f"Total rows: {len(df)}")
print(f"Rows with verbal response time: {df['verbal_response_time'].notna().sum()}")
print(f"Unique participants: {df['participant'].nunique()}")
print("\nSample of results:")
print(df[['participant', 'frame', 'multiclass_label', 'VAD_binary', 'verbal_response_time', 'has_vrt']].head(20))

# Check VAD distribution
print(f"\nVAD distribution:")
print(df['VAD_binary'].value_counts())

# Check label distribution
print(f"\nLabel distribution:")
print(df['multiclass_label'].value_counts().sort_index())

# Save results
df.to_csv('all_participants_with_vrt.csv', index=False)
print("\nResults saved to 'all_participants_with_vrt.csv'")

print(df)
