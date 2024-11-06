import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_pca_df(df): 
    participant_frames_labels = df.iloc[:, :4]
    x = df.iloc[:, 4:]
    
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=0.90)
    principal_components = pca.fit_transform(x)
    print(principal_components.shape)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_' + str(i) for i in range(principal_components.shape[1])])
    
    principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

    return principal_df

def create_normalized_df(df):
    participant_frames_labels = df.iloc[:, :4]
    
    features = df.columns[4:]
    norm_df = df.copy()
    
    scaler = StandardScaler()
    norm_df[features] = scaler.fit_transform(norm_df[features])
    
    norm_df = pd.concat([participant_frames_labels, norm_df[features]], axis=1)

    return norm_df

df = pd.read_csv("../preprocessing/merged_features/all_participants_merged_correct.csv")
participant_frames_labels = df.iloc[:, :4]

df_pose = df.iloc[:, 4:28]

df_facial = df.iloc[:, 28:63]

df_audio = df.iloc[:, 63:]


df_pose = pd.concat([participant_frames_labels, df_pose], axis=1)
df_pose_norm = create_normalized_df(df_pose)
df_pose_norm_pca = create_pca_df(df_pose_norm)

df_pose.to_csv("all_participants_pose_features.csv", index=False)
df_pose_norm.to_csv("all_participants_pose_features_norm.csv", index=False)
df_pose_norm_pca.to_csv("all_participants_pose_features_norm_pca.csv", index=False)

df_facial = pd.concat([participant_frames_labels, df_facial], axis=1)
df_facial_norm = create_normalized_df(df_facial)
df_facial_norm_pca = create_pca_df(df_facial_norm)

df_facial.to_csv("all_participants_facial_features.csv", index=False)
df_facial_norm.to_csv("all_participants_facial_features_norm.csv", index=False)
df_facial_norm_pca.to_csv("all_participants_facial_features_norm_pca.csv", index=False)

df_audio = pd.concat([participant_frames_labels, df_audio], axis=1)
df_audio_norm = create_normalized_df(df_audio)
df_audio_norm_pca = create_pca_df(df_audio_norm)

df_audio.to_csv("all_participants_audio_features.csv", index=False)
df_audio_norm.to_csv("all_participants_audio_features_norm.csv", index=False)
df_audio_norm_pca.to_csv("all_participants_audio_features_norm_pca.csv", index=False)

df_pose_facial = pd.concat([participant_frames_labels, df_pose, df_facial], axis=1)
df_pose_facial_norm = create_normalized_df(df_pose_facial)
df_pose_facial_norm_pca = create_pca_df(df_pose_facial_norm)

df_pose_facial.to_csv("all_participants_pose_facial_features.csv", index=False)
df_pose_facial_norm.to_csv("all_participants_pose_facial_features_norm.csv", index=False)
df_pose_facial_norm_pca.to_csv("all_participants_pose_facial_features_norm_pca.csv", index=False)

df_pose_audio = pd.concat([participant_frames_labels, df_pose, df_audio], axis=1)
df_pose_audio_norm = create_normalized_df(df_pose_audio)
df_pose_audio_norm_pca = create_pca_df(df_pose_audio_norm)

df_pose_audio.to_csv("all_participants_pose_audio_features.csv", index=False)
df_pose_audio_norm.to_csv("all_participants_pose_audio_features_norm.csv", index=False)
df_pose_audio_norm_pca.to_csv("all_participants_pose_audio_features_norm_pca.csv", index=False)

df_facial_audio = pd.concat([participant_frames_labels, df_facial, df_audio], axis=1)
df_facial_audio_norm = create_normalized_df(df_facial_audio)
df_facial_audio_norm_pca = create_pca_df(df_facial_audio_norm)

df_facial_audio.to_csv("all_participants_facial_audio_features.csv", index=False)
df_facial_audio_norm.to_csv("all_participants_facial_audio_features_norm.csv", index=False)
df_facial_audio_norm_pca.to_csv("all_participants_facial_audio_features_norm_pca.csv", index=False)

