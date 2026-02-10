import pandas as pd
import numpy as np


def linear_interpolate(df, original_fps, new_fps):

    feature_columns = [col for col in df.columns if col not in ['frame', 'participant', 'binary_label', 'multiclass_label']]

    new_data = []

    for participant, g in df.groupby('participant'):
        g = g.sort_values('frame').reset_index(drop=True)
        idx = g.index

        t_original = g['frame'].values / original_fps
        t_new = np.arange(
            t_original[0],
            t_original[-1] + 1e-8,
            1 / new_fps
        )

        interpolated_data = {col: np.interp(t_new, t_original, g[col].values) for col in feature_columns}

        new_participant_df = pd.DataFrame(interpolated_data)

        nearest_idx = np.searchsorted(t_original, t_new, side='left')
        nearest_idx = np.clip(nearest_idx, 0, len(g) - 1)

        new_participant_df['participant'] = participant
        new_participant_df['frame'] = np.arange(len(t_new))
        new_participant_df['binary_label'] = g['binary_label'].values[nearest_idx]
        new_participant_df['multiclass_label'] = g['multiclass_label'].values[nearest_idx]

        new_participant_df = new_participant_df[['frame', 'participant', 'binary_label', 'multiclass_label'] + feature_columns]
        new_data.append(new_participant_df)

    new_df = (
        pd.concat(new_data, ignore_index=True)
          .sort_values(['participant', 'frame'])
          .reset_index(drop=True)
    )

    return new_df

df = pd.read_csv("preprocessing/full_features/all_participants_0_3.csv")
df_curated = pd.read_csv("preprocessing/curated_features/curated_features_dataset_v3.csv")

original_fps = 30
new_fps = 100

interpolated_df = linear_interpolate(df, original_fps, new_fps)
interpolated_curated_df = linear_interpolate(df_curated, original_fps, new_fps)

print(interpolated_df)
print(interpolated_curated_df)

interpolated_df.to_csv("preprocessing/interpolation/interpolated.csv", index=False)
interpolated_curated_df.to_csv("preprocessing/interpolation/curated_features_dataset_v3_interpolated.csv", index=False)