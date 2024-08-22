import pandas as pd

df = pd.read_csv('preprocessing/merged_features/all_participants_normalized.csv')

participant_frames_labels = df.iloc[:, :4]

delta_columns = df.filter(regex='delta$', axis=1)

face_audio_features = df.iloc[:, 36:]

combined_df = pd.concat([participant_frames_labels, delta_columns, face_audio_features], axis=1)

combined_df.to_csv('preprocessing/merged_features/all_participants_normalized_deltas.csv', index=False)
