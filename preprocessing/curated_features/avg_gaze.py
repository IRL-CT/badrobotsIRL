"""
take the average of the two eyes to reduce noise

df['gaze_x'] = (df['gaze_0_x'] + df['gaze_1_x']) / 2
df['gaze_y'] = (df['gaze_0_y'] + df['gaze_1_y']) / 2
df['gaze_z'] = (df['gaze_0_z'] + df['gaze_1_z']) / 2

"""

import numpy as np
import pandas as pd

df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

# Average gaze vector to reduce noise by combining both eyes as a single feature
df['gaze_x'] = (df['gaze_0_x'] + df['gaze_1_x']) / 2
df['gaze_y'] = (df['gaze_0_y'] + df['gaze_1_y']) / 2
df['gaze_z'] = (df['gaze_0_z'] + df['gaze_1_z']) / 2

print(df)

df.to_csv("all_participants_0_3_with_avg_gaze.csv", index=False)