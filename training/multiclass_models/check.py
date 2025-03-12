import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_normalized_df(df):
    if df.empty:
        raise ValueError("create_normalized_df: Input DataFrame is empty.")
    participant_frames_labels = df.iloc[:, :4]
    
    features = df.columns[4:]
    norm_df = df.copy()
    
    scaler = StandardScaler()
    norm_df[features] = scaler.fit_transform(norm_df[features])
    
    norm_df = pd.concat([participant_frames_labels, norm_df[features]], axis=1)

    return norm_df

df = pd.read_csv("../../preprocessing/full_features/all_participants_0_3.csv")
df_stats = pd.read_csv("../../preprocessing/stats_features/all_participants_stats_0_3.csv")
df_rf = pd.read_csv("../../preprocessing/rf_features/all_participants_rf_0_3_40.csv")

info = df.iloc[:, :4]
df_pose_index = df.iloc[:, 4:28]
df_facial_index = pd.concat([df.iloc[:, 28:63], df.iloc[:, 88:]], axis=1)
df_audio_index = df.iloc[:, 63:88]

df_facial_index_stats = df_stats.iloc[:, 4:30]
df_audio_index_stats = df_stats.iloc[:, 30:53]

df_facial_index_rf = df_rf.iloc[:, 38:]
df_pose_index_rf = df_rf.iloc[:, 4:28]
df_audio_index_rf = df_rf.iloc[:, 28:38]

modality_mapping = {
    "pose": pd.concat([info, df_pose_index], axis=1),
    "facial": pd.concat([info, df_facial_index], axis=1),
    "audio": pd.concat([info, df_audio_index], axis=1)
}

print(create_normalized_df(modality_mapping.get("pose")))