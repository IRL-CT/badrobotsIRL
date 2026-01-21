"""

can calculate using the gaze_angle_x and gaze_angle_y

df['gaze_angle_mag'] = np.sqrt(df['gaze_angle_x']**2 + df['gaze_angle_y']**2)

to tell us how far the gaze deviates from the center of gaze

"""

import numpy as np
import pandas as pd

df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

df['gaze_angle_mag'] = np.sqrt(df['gaze_angle_x']**2 + df['gaze_angle_y']**2)


print(df)

df.to_csv("all_participants_0_3_with_gaze_mag.csv", index=False)