import pandas as pd
import os

all_participant_data_csv = "../preprocessing/merged_features/all_participants_merged_correct.csv"
directory = "../preprocessing/final_features/face" 

features = ['frame',' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' gaze_angle_x', ' gaze_angle_y']
# gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y

dataframes = []

for file in os.listdir(directory):
    if file.endswith(".csv"):
        participant = participant_id = file.split('.')[0]
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath, usecols=features)
        df['participant'] = participant
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.sort_values(by=['participant', 'frame'], ascending=True, inplace=True)

print(combined_df)
combined_df.to_csv("gaze_features.csv")
all_participant_data = pd.read_csv(all_participant_data_csv)

merged_df = pd.merge(all_participant_data, combined_df, on=['participant', 'frame'], how='left') 

merged_df.to_csv("all_participants.csv", index=False)
print(merged_df)
