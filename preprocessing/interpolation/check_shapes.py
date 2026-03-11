import pandas as pd
df_v4 = pd.read_csv('curated_features_dataset_v4.csv')
pose = pd.read_csv('interpolated_pose_features.csv')
print(f'v4 shape: {df_v4.shape}')
print(f'pose shape: {pose.shape}')
if 'frame' in pose.columns:
    merged = pd.merge(df_v4, pose, on=['participant', 'frame'], how='inner')
    print(f'Inner join shape: {merged.shape}')
else:
    print('Pose has no frame column.')
