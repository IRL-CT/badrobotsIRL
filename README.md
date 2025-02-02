# BAD Robots IRL

Recognizing robot failure by analyzing human reactions and behaviors toward in-person robot failures.

The experiment involved a participant and a robot interacting or conversing with each other in a private room. The robot was controlled by a researcher and was engineered to create at least 3 errors.

The robot failure: not understanding the participant's order.

The robot failure was verbalized as "Sorry, I do not understand" and occurred at least 3 times. Afterward, the interaction ended with robot verbalizing "OK, I will call the researcher".

## Table of Contents

1. [Analysis of Human Reactions to Robot Failure](https://github.com/FAR-Lab/badrobotsIRL?tab=readme-ov-file#analysis-of-human-reactions-to-robot-failure)
2. [Feature Extraction](https://github.com/FAR-Lab/badrobotsIRL?tab=readme-ov-file#feature-extraction)
3. [Labels](https://github.com/FAR-Lab/badrobotsIRL?tab=readme-ov-file#labels)
4. [Participant Exclusion](https://github.com/FAR-Lab/badrobotsIRL?tab=readme-ov-file#participant-exclusion)
5. [Principal Component Analysis](https://github.com/FAR-Lab/badrobotsIRL/tree/main?tab=readme-ov-file#principal-component-analysis-pca)

## Analysis of Human Reactions to Robot Failure

See [HRI25_LBR](https://github.com/FAR-Lab/badrobotsIRL/tree/main/HRI25_LBR) for more details on the study and findings.

## Features

Feature extraction was performed on the participant to understand and analyze facial expressions, body movements, and speech that might convey underlying emotions during the human-robot interaction. After each feature extraction tool was applied to the sample's videos, resulting outputs were processed into readable forms which were then merged into one collective CSV file documenting all feature data per frame per participant.

### Feature Extraction

#### Facial Features

The OpenFace toolkit was used to detect facial landmarks and action units and required a video file path as input. The output consisted of a CSV file containing feature data per frame for the video.

The OpenFace toolkit can be found here: https://github.com/TadasBaltrusaitis/OpenFace

##### Feature Exclusion

Irrelevant features were excluded from the OpenFace output and this process was completed while merging all feature data. Facial feature exclusion is found in this Python script: [feature_merge.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/feature_merge.py). For the facial feature exclusion portion of the script, the CSV file path of the facial features for each participant and final facial feature list were required as input.

Final facial feature list (mainly action units):
```
facial_features = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r',  'AU45_r', 'AU01_c',
'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c',
'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c', 'gaze_0_x', 'gaze_0_y',
'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']
```
 
#### Pose Features & Estimation

The OpenPose toolkit and the BODY_25 model were used to obtain keypoints of the participant's body features and required video file path as input and a JSON file path to store the output. The JSON files were parsed and converted into CSV files listing pose features per frame for each video file with this script: [parse_openpose.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/parse_openpose.py). 

The following was executed in the command line interface to return a JSON file of 25 keypoints per frame:
```
bin\OpenPoseDemo.exe --video {input_video_file_path} --write_video {output_file_path} --write_json {output_file_path} 
```

##### Feature Exclusion

Only upper body keypoints were relevant to the video dataset so the lower body keypoints (mid hip, right hip, left hip, right knee, left knee, right ankle, left ankle, right big toe, left big toe, right small toe, left small toe, right heel, left heel) were removed during preprocessing. The Python script used to exclude lower body features is found here: [feature_exclusion.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/feature_exclusion.py). Required input for the script includes the directory path to the directory holding all participant CSV files and a CSV file path for each participant to store output.

Final pose feature list: 
```
pose_features = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rightshoulder_x', 'rightshoulder_y',
'leftshoulder_x', 'leftshoulder_y', 'rightelbow_x', 'rightelbow_y', 'leftelbow_x', 'leftelbow_y',
'rightwrist_x', 'rightwrist_y', 'leftwrist_x', 'leftwrist_y','righteye_x', 'righteye_y',
'lefteye_x', 'lefteye_y', 'rightear_x', 'rightear_y', 'leftear_x', 'leftear_y']
```

The OpenPose toolkit can be found here: https://github.com/CMU-Perceptual-Computing-Lab/openpose

##### Body Keypoint Delta Calculations

In addition to the original features produced with OpenPose, another column titled "[original feature name]_delta" was appended and included the change in value from the previous frame's original feature value to the current frame's original feature value. The Python script used for updating the CSV with delta values is found here: [features_delta_calculations.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/features_delta_calculations.py). Required input to obtain delta calculations for each participant includes the CSV file path of pose features for the participant and a CSV file path to store the output or additional delta columns and values.

#### Audio Features

The openSMILE toolkit and the eGeMAPSv02 configuration were used to obtain audio features from the interaction between the participant and the robot and required an audio file path as input and a CSV file path as output. The following Python code was executed to return a CSV file of audio features:
```python
import opensmile
import os
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
file = f"{input_audio_file_path}"
y = smile.process_file(file)
y.to_csv(f"{output_file_path}")
```
The script can also be found here: [audio_extraction.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/audio_extraction.py). Required inputs for the script included a directory of all participant audio files and a CSV file path for each participant to store output.

Final audio feature list:
```
audio_features = ['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3', 'slope0-500_sma3', 'slope500-1500_sma3',
'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3', 'F0semitoneFrom27.5Hz_sma3nz',
'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz,HNRdBACF_sma3nz', 'logRelF0-H1-H2_sma3nz', 'logRelF0-H1-A3_sma3nz',
'F1frequency_sma3nz', 'F1bandwidth_sma3nz', 'F1amplitudeLogRelF0_sma3nz', 'F2frequency_sma3nz', 'F2bandwidth_sma3nz',
'F2amplitudeLogRelF0_sma3nz', 'F3frequency_sma3nz', 'F3bandwidth_sma3nz', 'F3amplitudeLogRelF0_sma3nz']
```

Only audio segments (partitioned via speaker diarization) spoken by the participant were relevant to the study, other segments were removed during preprocessing.

The openSMILE toolkit can be found here: https://audeering.github.io/opensmile/about.html or https://github.com/audeering/opensmile/

##### Speaker Diarization

Speaker diarization was used to identify timestamps indicating when the participant and the robot were speaking.

pyannote speaker diarization toolkit was used to extract timestamps. The following Python code was executed to return an RTTM file of speaker timestamps from an audio file.

```python
from pyannote.audio import Pipeline
import torch
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=f"{huggingface_authentication_token}")
diarization = pipeline(f"{input_audio_file_path}")
for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    with open(f"{output_file_path}", "w") as rttm:
        diarization.write_rttm(rttm)
```
The output of the speaker diarization was an RTTM file. This was converted into a CSV file and "overlapped" with the openSMILE audio extraction CSV outputs to remove the timestamps at which other speakers were speaking.

pyannote speaker diarization toolkit can be found here: https://huggingface.co/pyannote/speaker-diarization-3.1 or https://github.com/pyannote/pyannote-audio

Once the timestamps for each speaker were extracted, audio features corresponding to the participant's speech were retained, while those corresponding to other speakers' speech were removed. The Python script used to achieve this is found here: [filter_audio_features.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/filter_audio_features.py). Required inputs for the script to filter features for a single participant include the CSV file of openSMILE audio extracted features, the CSV file of timestamps produced from speaker diarization, and a CSV file path to store the output.

##### Timestamp to Frame Conversion

The CSV files of features achieved from openSMILE used start and end timestamps in the format "0 days 00:00:00.00", and each row included audio features for 0.02 seconds (between the start and end timestamps). However, this study is interested in utilizing frames instead of timestamps. Therefore, start timestamps were converted into frames. For multiple rows with the same frame number, the average feature value was calculated and used as the frame's feature value for each feature. The Python script used to convert timestamps to frames and process features is found here: [audio_features_frames.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/audio_features_frames.py). Required inputs to obtain frames included a directory of all participant CSV files of filtered audio features and a CSV file path for each participant to store the new output CSV.

### Feature Merge

After facial, pose, and audio features were processed to include relevant features per frame per participant, all features were merged into one CSV file representing the features per frame of a single participant. The Python script for merging features for each participant is found here: [feature_merge.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/feature_merge.py)

All participant's features were merged into a collective CSV file containing all rows from each participant's merged features data. The Python script for merging all participant feature data is found here: [feature_all_participants.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/feature_all_participants.py).

Features were checked for NaN and inf values and then normalized for model training. The Python script for checking feature values is found here: [check_features.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/check_features.py) and for normalizing features is found here [normalization.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/normalization.py)

### Feature Selection

There were three feature sets used when training our model. 

#### Full Feature Set

The full feature set contains all final facial, pose, and audio features listed previously.

#### Stats Feature Set

Features kept: [stats_features](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/stats/stats_features_ttest_full.csv)

#### Random Forest (RF) Feature Set

Selected features if feature drop > 40%

Features kept: [rf_features](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/rf/rf_features_selected_40.csv)

## Labels

Two types of labeling methods were used: binary labeling and multiclass labeling. A label was applied to each frame of the video of the interaction between the participant and the robot. Labels were initially appended to the pose feature CSV files.

The Python script used for efficient labeling are found here: [features_create_labels_csvs.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/features_create_labels_csvs.py). Required inputs for the script to store binary and multiclass labels for each participant included a CSV file to store the features and labels, the frame numbers for when binary labels should be added, and the frame numbers for when multiclass labels should be added. Additionally, an input directory path was required for the location to store CSV files of features and labels for each participant.

### Binary Labeling

- "0" labeled frames from the beginning of the video to the first "Sorry, I do not understand" error and after "OK, I will call the researcher" to the end of the video.
- "1" labeled frames from the first "Sorry, I do not understand" error to "OK, I will call the researcher".

### Multiclass Labeling

- "0" labeled frames from the beginning of the video to the first "Sorry, I do not understand" error and after "OK, I will call the researcher" to the end of the video.
- "1" labeled frames from the first "Sorry, I do not understand" error to the second "Sorry, I do not understand".
- "2" labeled frames from the second "Sorry, I do not understand" error to the third "Sorry, I do not understand".
- "3" labeled frames from the third "Sorry, I do not understand" error to "OK, I will call the researcher".

The same labeling procedure was be used for additional errors.

### Label Analysis

The following charts contain the amount and percentages of labels noted per participant.

#### Binary Labeling
| participant  | Label "0" Count | Label "0" Percentage | Label "1" Count | Label "1" Percentage | Total Number of Labels|
|-----------|---------------|--------------------|---------------|--------------------|-------|
| 2         | 569           | 48.9%              | 595           | 51.1%              | 1164  |
| 4         | 601           | 61.2%              | 381           | 38.8%              | 982   |
| 5         | 575           | 49.6%              | 584           | 50.4%              | 1159  |
| 6         | 619           | 51.9%              | 573           | 48.1%              | 1192  |
| 7         | 609           | 53.8%              | 524           | 46.2%              | 1133  |
| 8         | 606           | 59.8%              | 408           | 40.2%              | 1014  |
| 9         | 582           | 55.8%              | 461           | 44.2%              | 1043  |
| 10        | 589           | 48.3%              | 630           | 51.7%              | 1219  |
| 11        | 588           | 55.2%              | 477           | 44.8%              | 1065  |
| 12        | 615           | 45.8%              | 728           | 54.2%              | 1343  |
| 14        | 571           | 54.9%              | 469           | 45.1%              | 1040  |
| 16        | 577           | 43.0%              | 764           | 57.0%              | 1341  |
| 17        | 585           | 51.7%              | 546           | 48.3%              | 1131  |
| 18        | 577           | 45.1%              | 703           | 54.9%              | 1280  |
| 19        | 579           | 49.8%              | 583           | 50.2%              | 1162  |
| 20        | 595           | 44.4%              | 746           | 55.6%              | 1341  |
| 21        | 583           | 49.0%              | 607           | 51.0%              | 1190  |
| 22        | 571           | 52.1%              | 526           | 47.9%              | 1097  |
| 23        | 596           | 50.1%              | 593           | 49.9%              | 1189  |
| 25        | 577           | 50.9%              | 556           | 49.1%              | 1133  |
| 26        | 519           | 45.1%              | 631           | 54.9%              | 1150  |
| 27        | 608           | 39.2%              | 942           | 60.8%              | 1550  |
| 28        | 600           | 53.1%              | 531           | 46.9%              | 1131  |
| 29        | 517           | 32.6%              | 1069          | 67.4%              | 1586  |

- Average percentage of "0" labels: 49.6%
- Average percentage of "1" labels: 50.4%

#### Multiclass Labeling
| participant  | Label "0" Count | Label "0" Percentage | Label "1" Count | Label "1" Percentage | Label "2" Count | Label "2" Percentage | Label "3" Count | Label "3" Percentage | Total Number of Labels|
|-----------|---------------|--------------------|---------------|--------------------|---------------|--------------------|---------------|--------------------|-------|
| 2         | 569           | 48.9%              | 197           | 16.9%              | 202           | 17.4%              | 196           | 16.8%              | 1164  |
| 4         | 601           | 61.2%              | 132           | 13.4%              | 129           | 13.1%              | 120           | 12.2%              | 982   |
| 5         | 575           | 49.6%              | 221           | 19.1%              | 130           | 11.2%              | 233           | 20.1%              | 1159  |
| 6         | 619           | 51.9%              | 166           | 13.9%              | 147           | 12.3%              | 260           | 21.8%              | 1192  |
| 7         | 609           | 53.8%              | 159           | 14.0%              | 194           | 17.1%              | 171           | 15.1%              | 1133  |
| 8         | 606           | 59.8%              | 143           | 14.1%              | 136           | 13.4%              | 129           | 12.7%              | 1014  |
| 9         | 582           | 55.8%              | 136           | 13.0%              | 164           | 15.7%              | 161           | 15.4%              | 1043  |
| 10        | 589           | 48.3%              | 166           | 13.6%              | 275           | 22.6%              | 189           | 15.5%              | 1219  |
| 11        | 588           | 55.2%              | 140           | 13.1%              | 206           | 19.3%              | 131           | 12.3%              | 1065  |
| 12        | 615           | 45.8%              | 166           | 12.4%              | 165           | 12.3%              | 397           | 29.6%              | 1343  |
| 14        | 571           | 54.9%              | 147           | 14.1%              | 160           | 15.4%              | 162           | 15.6%              | 1040  |
| 16        | 577           | 43.0%              | 165           | 12.3%              | 263           | 19.6%              | 336           | 25.1%              | 1341  |
| 17        | 585           | 51.7%              | 155           | 13.7%              | 195           | 17.2%              | 196           | 17.3%              | 1131  |
| 18        | 577           | 45.1%              | 137           | 10.7%              | 187           | 14.6%              | 379           | 29.6%              | 1280  |
| 19        | 579           | 49.8%              | 145           | 12.5%              | 212           | 18.2%              | 226           | 19.4%              | 1162  |
| 20        | 595           | 44.4%              | 168           | 12.5%              | 276           | 20.6%              | 302           | 22.5%              | 1341  |
| 21        | 583           | 49.0%              | 156           | 13.1%              | 205           | 17.2%              | 246           | 20.7%              | 1190  |
| 22        | 571           | 52.1%              | 158           | 14.4%              | 162           | 14.8%              | 206           | 18.8%              | 1097  |
| 23        | 596           | 50.1%              | 130           | 10.9%              | 341           | 28.7%              | 122           | 10.3%              | 1189  |
| 25        | 577           | 50.9%              | 137           | 12.1%              | 218           | 19.2%              | 201           | 17.7%              | 1133  |
| 26        | 519           | 45.1%              | 170           | 14.8%              | 193           | 16.8%              | 268           | 23.3%              | 1150  |
| 27        | 608           | 39.2%              | 154           | 9.9%               | 298           | 19.2%              | 490           | 31.6%              | 1550  |
| 28        | 600           | 53.1%              | 161           | 14.2%              | 234           | 20.7%              | 136           | 12.0%              | 1131  |

Participant 29's interaction consisted of 5 errors. Therefore, it contained 2 additional labels "4" and "5".
| participant  | Label "0" Count | Label "0" Percentage | Label "1" Count | Label "1" Percentage | Label "2" Count | Label "2" Percentage | Label "3" Count | Label "3" Percentage | Label "4" Count | Label "4" Percentage | Label "5" Count | Label "5" Percentage | Total Number of Labels|
|-----------|---------------|--------------------|---------------|--------------------|---------------|--------------------|---------------|--------------------|---------------|--------------------|---------------|--------------------|-------|
| 29        | 511           | 32.2%              | 233           | 14.7%              | 167           | 10.5%              | 191           | 12.0%              | 223           | 14.1%              | 261           | 16.5%              | 1586  |


- Average percentage of "0" label: 49.5%
- Average percentage of "1" label: 12.7%
- Average percentage of "2" label: 16.6%
- Average percentage of "3" label: 19.0%
- Average percentage of additional labels: 2.2%

The Python script that assisted with the label analysis is found here: [label_analysis.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/preprocessing/label_analysis.py)

## Participant Exclusion

Participants were excluded based on the following reasons:
- Failed protocol resulting in no reaction to failures
- Distractions not involved in the experiment resulting in no reaction to failures
- Feature extraction compound confidence scores below 0.50.

Final number of participants: 24.

## Principal Component Analysis (PCA)

PCA is a method used to reduce the number of variables in a large dataset by retaining patterns in the data. PCA was conducted on the dataset of 84 features containing facial, pose, and audio features. The short script below was used to retain 90% of the variance and apply the PCA to the dataset.

```python
participant_frames_labels = df.iloc[:, :4]
x = df.iloc[:, 4:]
x = StandardScaler().fit_transform(x.values)
pca = PCA()
principal_components = pca.fit_transform(x)
print(principal_components.shape)

pca = PCA(n_components=0.90)
principal_components = pca.fit_transform(x)
print(principal_components.shape)

principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)
```
The script was embedded into the create_data_splits_pca.py method in [create_data_splits.py](https://github.com/FAR-Lab/badrobotsIRL/blob/main/training/create_data_splits.py).

The resulting dataframe consisted of 41 principal components.

Running PCA on pose, facial, and audio features separately yielded 13, 24, and 7 principal components, respectively.
