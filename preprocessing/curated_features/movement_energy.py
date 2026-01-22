"""
could do this just for head or for all keypoints on the head
sum(|[body keypoint]_x_delta| + |[body keypoint]_y_delta|)

"""

import numpy as np
import pandas as pd

df = pd.read_csv("preprocessing/curated_features/all_participants_0_3.csv")

head_kps = ['nose', 'neck', 'righteye', 'lefteye', 'rightear', 'leftear']

df['head_movement_energy'] = 0.0

for kp in head_kps:
    dx = df[f'{kp}_x_delta']
    dy = df[f'{kp}_y_delta']
    df['head_movement_energy'] += dx**2 + dy**2

print(df)