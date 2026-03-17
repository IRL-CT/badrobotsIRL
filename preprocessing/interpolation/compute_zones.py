import pandas as pd
import numpy as np
import os

INPUT_PATH = 'interpolated_pose_features.csv'
CURATED_PATH = 'curated_features_v3_interpolated.csv'
OUTPUT_PATH = 'curated_features_zones.csv'

print(f"Loading '{INPUT_PATH}'...")
pose_df = pd.read_csv(INPUT_PATH)

print(f"Loading '{CURATED_PATH}'...")
curated_df = pd.read_csv(CURATED_PATH)

# Merge to ensure we only process the exact rows/frames in curated
print("Merging datasets on participant and frame...")
merged_df = pd.merge(curated_df, pose_df, on=['participant', 'frame'], how='inner')

print("Loading 'allparticipants_100fps.csv' for raw gaze angles and Action Units...")
raw_df = pd.read_csv('allparticipants_100fps.csv')
raw_df = raw_df[~raw_df['participant'].isin(['p3nodbot', 'p24nodbot'])].reset_index(drop=True)
# Keep only the rows that match our curated interpolated dataset
raw_merged = pd.merge(merged_df[['participant', 'frame']], raw_df, on=['participant', 'frame'], how='inner')

# 1. Head Zones (From OpenFace gaze angles in raw_merged)
# Converting gaze_angle_x and gaze_angle_y from radians to degrees
raw_merged['gaze_yaw_deg'] = raw_merged['gaze_angle_x'] * (180.0 / np.pi)
raw_merged['gaze_pitch_deg'] = raw_merged['gaze_angle_y'] * (180.0 / np.pi)

def assign_zone(val, thresholds):
    if pd.isna(val): return 0
    if -thresholds[0] <= val <= thresholds[0]: return 1
    elif val < -thresholds[0]: return 2
    else: return 3

merged_df['zone_head_heading'] = raw_merged['gaze_yaw_deg'].apply(lambda x: assign_zone(x, (15, 15)))
merged_df['zone_head_pitch'] = raw_merged['gaze_pitch_deg'].apply(lambda x: assign_zone(x, (15, 15)))

# 2. Trunk Tilt (using 2D shoulders)
def calculate_shoulder_tilt_2d(row):
    lx, ly = row['leftshoulder_x'], row['leftshoulder_y']
    rx, ry = row['rightshoulder_x'], row['rightshoulder_y']
    if pd.isna(lx) or pd.isna(rx) or lx == 0 or rx == 0: return np.nan
    dx, dy = lx - rx, ly - ry
    if dx == 0: return 0 if dy == 0 else 90
    return np.degrees(np.arctan2(dy, dx))

merged_df['trunk_tilt_proxy_deg'] = merged_df.apply(calculate_shoulder_tilt_2d, axis=1)
merged_df['zone_trunk_tilt'] = merged_df['trunk_tilt_proxy_deg'].apply(lambda x: assign_zone(x, (12, 12)))

# 3. Facial Expression Zones
au_c_cols = [c for c in raw_merged.columns if c.startswith('AU') and c.endswith('_c')]
raw_merged['mean_au_confidence'] = raw_merged[au_c_cols].mean(axis=1)

def assign_fe_zone(val):
    if pd.isna(val): return 0
    if val <= 0.5: return 1
    elif val <= 0.75: return 2
    else: return 3

merged_df['zone_facial_expression'] = raw_merged['mean_au_confidence'].apply(assign_fe_zone)

# We only want to keep the original v4 columns + the new zones
v4_cols = list(curated_df.columns)
new_zone_cols = [
    'zone_head_heading', 'zone_head_pitch', 
    'zone_trunk_tilt', 
    'zone_facial_expression'
]
final_df = merged_df[v4_cols + new_zone_cols]

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH}")
print(f"Shape: {final_df.shape}")
print(final_df.head())
# To replicate the paper, you would compute aggregation statistics over 0.5s windows. For RNNs, these frame-level zones are sufficient
