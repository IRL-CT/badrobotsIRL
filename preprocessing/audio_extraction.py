import opensmile
import os

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

directory = "./nodbot_wavs/"

files = os.listdir(directory)

for file in files:
    y = smile.process_file(directory+file)
    y.to_csv(f"./nodbot_audio_features/{file}_audio_features.csv")
