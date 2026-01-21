"""
could do this just for head or for all
sum(|[body keypoint]_x_delta| + |[body keypoint]_y_delta|)

"""

import numpy as np
import pandas as pd

df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

keypoints = ['nose',]

for kp in keypoints:
    df[f'{kp}_x_delta'] = df[f'{kp}_x'].diff()
    df[f'{kp}_y_delta'] = df[f'{kp}_y'].diff()
    df[f'{kp}_movement_energy'] = df[f'{kp}_x_delta'].abs() + df[f'{kp}_y_delta'].abs()

df.to_csv("all_participants_0_3_with_movement_energy.csv", index=False)