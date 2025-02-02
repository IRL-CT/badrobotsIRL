import pandas as pd
import matplotlib.pyplot as plt

# config: {'dataset': 'reg', 'feature_set_tag': 'full', 'max_depth': 10, 'modality': 'pose_audio', 'n_estimators': 700, 'sequence_length': 60}

# sorted_feature_importance = {' gaze_0_x': 0.0934678107680719, ' gaze_angle_x': 0.09205608815235256, ' gaze_0_z': 0.08945923141801461, ' gaze_1_x': 0.07717008921231835, ' gaze_1_z': 0.07489642580139956, ' gaze_1_y': 0.05762968653486568, ' gaze_angle_y': 0.05665992794155753, ' gaze_0_y': 0.04574451662721789, 'rightwrist_x_delta': 0.032715717174726586, 'rightwrist_y_delta': 0.03168249504282124, 'righteye_x_delta': 0.023323216396240288, 'nose_x_delta': 0.01902331284840967, 'rightelbow_x_delta': 0.017893654252662306, 'leftwrist_x_delta': 0.017854977945023092, 'lefteye_x_delta': 0.01710334290045413, 'leftwrist_y_delta': 0.016607431577211958, 'righteye_y_delta': 0.01274176456860272, 'nose_y_delta': 0.012509596219323038, 'lefteye_y_delta': 0.012273781881492085, 'leftear_x_delta': 0.010250917653578879, 'leftear_y_delta': 0.00968113431526955, 'rightear_y_delta': 0.00903537257252873, 'leftshoulder_x_delta': 0.008592058310500333, 'rightear_x_delta': 0.008589196158966228, 'Loudness_sma3': 0.008508735262623838, 'leftelbow_x_delta': 0.008128193411007897, 'spectralFlux_sma3': 0.007753620388874243, 'rightelbow_y_delta': 0.007660221305741616, 'leftelbow_y_delta': 0.007397109318156248, 'F2bandwidth_sma3nz': 0.007360531872791136, 'F2frequency_sma3nz': 0.007149718689873937, 'F3frequency_sma3nz': 0.007034615039310135, 'F1bandwidth_sma3nz': 0.006826888019229163, 'F1frequency_sma3nz': 0.006659871796300884, 'neck_x_delta': 0.006611675349416953, 'leftshoulder_y_delta': 0.0062734848828473935, 'rightshoulder_y_delta': 0.00604479931159928, 'neck_y_delta': 0.00577888801389634, 'F3bandwidth_sma3nz': 0.005696432604928998, 'rightshoulder_x_delta': 0.0052636592890351625, 'F0semitoneFrom27.5Hz_sma3nz': 0.005107156716755595, 'hammarbergIndex_sma3': 0.00470901251142633, 'F3amplitudeLogRelF0_sma3nz': 0.004362121539991881, 'alphaRatio_sma3': 0.0030878969823307546, 'F2amplitudeLogRelF0_sma3nz': 0.0025176384656811724, 'mfcc3_sma3': 0.002503816540873007, 'mfcc1_sma3': 0.0024604270052123126, 'slope0-500_sma3': 0.002191604642571896, 'logRelF0-H1-A3_sma3nz': 0.0020085247411951884, 'jitterLocal_sma3nz': 0.0020084643131262777, 'slope500-1500_sma3': 0.001944264058291286, 'HNRdBACF_sma3nz': 0.001877996796154912, 'mfcc4_sma3': 0.0018089769640084973, 'shimmerLocaldB_sma3nz': 0.0017124181116974571, 'F1amplitudeLogRelF0_sma3nz': 0.0017069126845952118, 'mfcc2_sma3': 0.001545928941392508, 'logRelF0-H1-H2_sma3nz': 0.0013366481554535933}
# feature_importance_df = pd.DataFrame.from_dict(sorted_feature_importance, orient='index')
# feature_importance_df.to_csv(f"feature_importances.csv")
# fig = plt.figure(figsize=(20, 10))
# plt.bar(sorted_feature_importance.keys(), sorted_feature_importance.values())
# plt.show()
# fig.savefig(f"feature_importances.png")

# cutoff at 30% drop --> if feature drop % from previous is over 30%, cut off that feature and all features with lesser importance

raw_data = '../preprocessing/merged_features/all_participants_0_3.csv'
data = pd.read_csv(raw_data)
features_rf = pd.read_csv('../preprocessing/feature_importances.csv')

features_rf['importance_drop'] = features_rf['importance'].pct_change(periods=-1).abs()
features_rf.to_csv("feature_importance_percentage_drop.csv")

print(features_rf)
cutoff_index = features_rf[features_rf['importance_drop'] > 0.4].index.min()

if pd.notna(cutoff_index):
    cutoff_feature = features_rf.iloc[cutoff_index]['feature']
    print(f"cutoff feature > 0.4 : {cutoff_feature}")
    features_to_keep = features_rf.iloc[:cutoff_index]['feature'].tolist()
else:
    features_to_keep = features_rf['feature'].tolist()

columns_in_data = data.columns.tolist()
non_feature_columns = ['frame', 'participant', 'binary_label', 'multiclass_label']
feature_columns_in_data = [col for col in columns_in_data if col not in non_feature_columns]
columns_to_select = non_feature_columns + [feature for feature in feature_columns_in_data if feature in features_to_keep]

filtered_df = data[columns_to_select]

print(features_to_keep)
print(filtered_df)

filtered_df.to_csv('../preprocessing/all_participants_rf_0_3_40.csv', index=False)