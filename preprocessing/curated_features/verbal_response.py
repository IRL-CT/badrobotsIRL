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

df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

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
    g = g.sort_index()

    labels = g['multiclass_label']
    vad = g['VAD_binary']

    # find error onsets

    error_onsets = g.index[
        labels == labels.shift(1) + 1
    ]

    # compute verbal response time for each error

    for error_frame in error_onsets:
        error_label = labels.loc[error_frame]

        # frames belonging to this error segment
        error_segment = g.index[
            labels == error_label
        ]

        # find first frame with voice activity after error onset
        vad_window = vad.loc[error_segment]
        speech_frames = vad_window[vad_window == 1]

        if len(speech_frames) == 0:
            continue  # participant never verbally responded

        verbal_frame = speech_frames.index[0]

        # response time in frames
        verbal_response_time = verbal_frame - error_frame

        # assign the same response time to all frames
        # belonging to this error
        df.loc[
            error_segment,
            'verbal_response_time'
        ] = verbal_response_time


df['has_vrt'] = df['verbal_response_time'].notna().astype(int)

print(df)

df.to_csv("all_participants_0_3_with_vrt.csv", index=False)
