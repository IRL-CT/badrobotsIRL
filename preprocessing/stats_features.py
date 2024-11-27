import pandas as pd

raw_data = '../preprocessing/merged_features/all_participants_merged_correct.csv'
normalized_data = '../preprocessing/merged_features/all_participants_merged_correct_normalized.csv'
normalized_pca_data = '../preprocessing/merged_features/all_participants_merged_correct_normalized_principal.csv'

data = pd.read_csv(normalized_pca_data)
features_stats = pd.read_csv('../preprocessing/stats/stats_features_ttest_full.csv')

features_to_keep = features_stats['feature'].tolist()

columns_in_data = data.columns.tolist()
non_feature_columns = ['frame', 'participant', 'binary_label', 'multiclass_label']

feature_columns_in_data = [col for col in columns_in_data if col not in non_feature_columns]

columns_to_select = non_feature_columns + [feature for feature in feature_columns_in_data if feature in features_to_keep]

filtered_df = data[columns_to_select]

filtered_df.to_csv('../preprocessing/stats_features/all_participants_merged_correct_normalized_principal.csv', index=False)
