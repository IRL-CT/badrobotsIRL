"""

the difference between gaze vectors in previous frame and gaze vectors in current frame (like distance formula)

df['gaze_shift'] = np.sqrt(
    (df['gaze_x'].diff())**2 +
    (df['gaze_y'].diff())**2 +
    (df['gaze_z'].diff())**2
)

how much the gaze vector moves from the previous frame to the current frame (velocity of gaze)

"""

import numpy as np
import pandas as pd

df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

df['gaze_shift'] = np.nan

for participant, g in df.groupby('participant'):
    idx = g.index
    df.loc[idx, 'gaze_shift'] = np.sqrt(
        g['gaze_x'].diff()**2 +
        g['gaze_y'].diff()**2 +
        g['gaze_z'].diff()**2
    )

print(df)
