import pandas as pd
import matplotlib.pyplot as plt

# config: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
'''
sorted_feature_importance = {
                    "gaze_0_z": 0.053123,
                    "gaze_angle_x": 0.045321,
                    "gaze_0_x": 0.044280,
                    "AU10_r": 0.042298,
                    "gaze_1_z": 0.037489,
                    "AU06_r": 0.034638,
                    "AU12_r": 0.034578,
                   "gaze_1_x": 0.033643,
                     "AU20_r": 0.031100,
                     "AU25_r": 0.029510,
               "gaze_angle_y": 0.028523,
                     "AU04_r": 0.028517,
                   "gaze_1_y": 0.028384,
                     "AU15_r": 0.027472,
                     "AU09_r": 0.026516,
                   "gaze_0_y": 0.025261,
                     "AU14_r": 0.023301,
                     "AU17_r": 0.022754,
                     "AU07_r": 0.020920,
                     "AU01_r": 0.020713,
                     "AU02_r": 0.020392,
                     "AU26_r": 0.019713,
                     "AU23_r": 0.014268,
                     "AU45_r": 0.012195,
                     "AU23_c": 0.010358,
         "rightwrist_x_delta": 0.010124,
           "righteye_x_delta": 0.010065,
         "rightwrist_y_delta": 0.009779,
                     "AU14_c": 0.009390,
               "nose_x_delta": 0.009371,
                     "AU05_r": 0.009277,
            "lefteye_x_delta": 0.009012,
            "lefteye_y_delta": 0.008293,
                     "AU10_c": 0.008202,
               "nose_y_delta": 0.007760,
                     "AU02_c": 0.007367,
                     "AU01_c": 0.007301,
          "leftwrist_x_delta": 0.006629,
          "leftwrist_y_delta": 0.006408,
                     "AU06_c": 0.006403,
                     "AU12_c": 0.006237,
                     "AU17_c": 0.006232,
           "righteye_y_delta": 0.006161,
                     "AU15_c": 0.006134,
                     "AU09_c": 0.005873,
           "rightear_x_delta": 0.005616,
                     "AU04_c": 0.005547,
          "spectralFlux_sma3": 0.005062,
            "leftear_x_delta": 0.004926,
         "rightelbow_x_delta": 0.004892,
                     "AU26_c": 0.004563,
                     "AU45_c": 0.004495,
                     "AU25_c": 0.004457,
                     "AU05_c": 0.004413,
                     "AU20_c": 0.004279,
         "F3frequency_sma3nz": 0.004274,
            "leftear_y_delta": 0.003847,
                     "AU07_c": 0.003726,
              "Loudness_sma3": 0.003724,
           "rightear_y_delta": 0.003624,
         "F1frequency_sma3nz": 0.003526,
          "leftelbow_x_delta": 0.003437,
         "F1bandwidth_sma3nz": 0.003272,
         "rightelbow_y_delta": 0.003245,
               "neck_x_delta": 0.003194,
       "leftshoulder_x_delta": 0.002857,
          "leftelbow_y_delta": 0.002855,
      "rightshoulder_x_delta": 0.002795,
         "F2frequency_sma3nz": 0.002629,
"F0semitoneFrom27.5Hz_sma3nz": 0.002588,
               "neck_y_delta": 0.002562,
 "F2amplitudeLogRelF0_sma3nz": 0.002536,
         "F3bandwidth_sma3nz": 0.002511,
       "leftshoulder_y_delta": 0.002495,
      "rightshoulder_y_delta": 0.002257,
                 "mfcc3_sma3": 0.001965,
         "F2bandwidth_sma3nz": 0.001923,
       "hammarbergIndex_sma3": 0.001887,
            "alphaRatio_sma3": 0.001569,
 "F3amplitudeLogRelF0_sma3nz": 0.001555,
         "slope500-1500_sma3": 0.001266,
                 "mfcc1_sma3": 0.001218,
            "slope0-500_sma3": 0.000967,
                     "AU28_c": 0.000939,
                 "mfcc2_sma3": 0.000826,
      "shimmerLocaldB_sma3nz":  0.000822,
      "logRelF0-H1-A3_sma3nz": 0.000794,
                 "mfcc4_sma3": 0.000747,
 "F1amplitudeLogRelF0_sma3nz": 0.000661,
            "HNRdBACF_sma3nz": 0.000554,
      "logRelF0-H1-H2_sma3nz": 0.000445,
         "jitterLocal_sma3nz":  0.000370
}
feature_importance_df = pd.DataFrame.from_dict(sorted_feature_importance, orient='index')
feature_importance_df.to_csv(f"feature_importances.csv")
fig = plt.figure(figsize=(20, 10))
plt.bar(sorted_feature_importance.keys(), sorted_feature_importance.values())
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
fig.savefig(f"feature_importances.png")
'''
# cutoff at 30% drop --> if feature drop % from previous is over 30%, cut off that feature and all features with lesser importance

raw_data = '../preprocessing/interpolation/allparticipants_100fps.csv'
data = pd.read_csv(raw_data)
features_rf = pd.read_csv('../preprocessing/feature_importances.csv')

features_rf['importance_drop'] = features_rf['importance'].pct_change(periods=-1).abs()
features_rf.to_csv("feature_importance_percentage_drop.csv")

print(features_rf)
cutoff_index = features_rf[features_rf['importance_drop'] > 0.3].index.min()

if pd.notna(cutoff_index):
    cutoff_feature = features_rf.iloc[cutoff_index]['feature']
    print(f"cutoff feature > 0.3 : after {cutoff_feature}")
    features_to_keep = features_rf.iloc[:cutoff_index+1]['feature'].tolist()
else:
    features_to_keep = features_rf['feature'].tolist()

columns_in_data = data.columns.tolist()
non_feature_columns = ['frame', 'participant', 'binary_label', 'multiclass_label']
feature_columns_in_data = [col for col in columns_in_data if col not in non_feature_columns]
columns_to_select = non_feature_columns + [feature for feature in feature_columns_in_data if feature in features_to_keep]

filtered_df = data[columns_to_select]

print(features_to_keep)
print(data.columns)
print(filtered_df.columns)

filtered_df.to_csv('../preprocessing/interpolation/rf_allparticipants_100fps.csv', index=False)