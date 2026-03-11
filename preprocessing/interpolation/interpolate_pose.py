import pandas as pd
import numpy as np
import os
import glob

# Paths
POSE_DIR = '../relevant_pose_features'
TARGET_DF_PATH = 'allparticipants_pose_100fps.csv'
OUTPUT_PATH = 'interpolated_pose_features.csv'

print("Loading reference dataset...")
ref_df = pd.read_csv(TARGET_DF_PATH)

# Get sorted participants from reference, excluding p3 and p24
target_participants = sorted([p for p in ref_df['participant'].unique() if p not in ['p3nodbot', 'p24nodbot']])

# Columns to interpolate
POSE_COLS = [
    'nose_x', 'nose_y', 'nose_c',
    'neck_x', 'neck_y', 'neck_c',
    'rightshoulder_x', 'rightshoulder_y', 'rightshoulder_c',
    'rightelbow_x', 'rightelbow_y', 'rightelbow_c',
    'rightwrist_x', 'rightwrist_y', 'rightwrist_c',
    'leftshoulder_x', 'leftshoulder_y', 'leftshoulder_c',
    'leftelbow_x', 'leftelbow_y', 'leftelbow_c',
    'leftwrist_x', 'leftwrist_y', 'leftwrist_c',
    'righteye_x', 'righteye_y', 'righteye_c',
    'lefteye_x', 'lefteye_y', 'lefteye_c',
    'rightear_x', 'rightear_y', 'rightear_c',
    'leftear_x', 'leftear_y', 'leftear_c'
]

all_interpolated = []

for participant in target_participants:
    print(f"Processing {participant}...")
    
    # Load corresponding pose file
    pose_file = os.path.join(POSE_DIR, f"{participant}_pose_features.csv")
    if not os.path.exists(pose_file):
        print(f"  Missing {pose_file}, skipping")
        continue
        
    pose_df = pd.read_csv(pose_file)
    pose_df.columns = pose_df.columns.str.strip()
    
    # Get reference rows for this participant to know exact target length
    ref_part = ref_df[ref_df['participant'] == participant]
    n_target = len(ref_part)
    n_video = len(pose_df)
    
    if n_video == 0:
        continue
        
    ratio = n_target / n_video
    
    # Create target positions (mapped to video frame indices)
    target_positions = np.arange(n_target) / ratio
    video_positions = np.arange(n_video).astype(float)
    
    # Create new dataframe
    interp_df = pd.DataFrame({'participant': [participant] * n_target})
    interp_df['frame'] = ref_part['frame'].values if 'frame' in ref_part.columns else np.arange(n_target)
    
    # Linearly interpolate each pose feature
    for col in POSE_COLS:
        if col in pose_df.columns:
            video_vals = pose_df[col].values.astype(float)
            interpolated = np.interp(target_positions, video_positions, video_vals)
            interp_df[col] = interpolated
            
    all_interpolated.append(interp_df)

print(f"Concatenating {len(all_interpolated)} participants...")
final_df = pd.concat(all_interpolated, ignore_index=True)

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
print(f"Shape: {final_df.shape}")
