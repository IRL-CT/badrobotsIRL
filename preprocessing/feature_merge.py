import pandas as pd

for i in range(25, 30):

    face_url = f"preprocessing/final_features/face/p{i}nodbot.csv"
    pose_url = f"preprocessing/final_features/pose/p{i}nodbot_pose_features_deltas_labels.csv"
    audio_url = f"preprocessing/final_features/audio_averaged/p{i}nodbot_audio.csv"

    face_df = pd.read_csv(face_url)
    vis_features = ['frame', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', 
                    ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', 
                    ' AU26_r',  ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', 
                    ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c',
                    ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
    relevant_face_df = face_df[vis_features]

    pose_df = pd.read_csv(pose_url)
    pose_df.drop(columns=["Unnamed: 0"], inplace=True)
    participant_frame = pose_df.iloc[:, :2]
    delta_columns = pose_df.filter(regex='delta$', axis=1)
    label_columns = pose_df.iloc[:, -2:]
    pose_df = pd.concat([participant_frame, delta_columns, label_columns], axis=1)

    audio_df = pd.read_csv(audio_url)

    face_pose_merged_df = pd.merge(pose_df, relevant_face_df, on="frame", how="inner")
    face_pose_audio_merged_df = pd.merge(face_pose_merged_df, audio_df, on="frame", how="inner")
    face_pose_audio_merged_df = face_pose_audio_merged_df.iloc[:, 0:]
    cols = list(face_pose_audio_merged_df.columns)
    cols.insert(0, cols.pop(cols.index('frame')))
    cols.insert(1, cols.pop(cols.index('participant')))
    cols.insert(2, cols.pop(cols.index('binary_label')))
    cols.insert(3, cols.pop(cols.index('multiclass_label')))
    face_pose_audio_merged_df = face_pose_audio_merged_df[cols]

    print(face_pose_audio_merged_df)

    face_pose_audio_merged_df.to_csv(f"preprocessing/merged_features/merged_features_correct/p{i}nodbot_all_features_correct.csv", index=False)
