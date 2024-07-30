from datetime import datetime
import pandas
import os

def time_to_frame(time_str, frame_rate=29.97):
    
    if pandas.isna(time_str):
        return None

    time_str = time_str.split(" ")[-1]

    total_seconds = 0

    if (len(time_str)) == 8:
        hr = time_str.split(":")[0]
        min = time_str.split(":")[1]
        sec = time_str.split(":")[2]
        total_seconds = float(hr) * 3600 + float(min) * 60 + float(sec)

    elif (len(time_str)) >8:
        microsec = time_str.split(".")[-1]
        hr = time_str.split(".")[0].split(":")[0]
        min = time_str.split(".")[0].split(":")[1]
        sec = time_str.split(".")[0].split(":")[2]
        microsec = time_str.split(".")[-1]
        total_seconds = float(hr) * 3600 + float(min) * 60 + float(sec) + float(microsec) / 1e6

    return int(total_seconds * frame_rate)


def create_frames_audio_csv(file, participant):

    df = pandas.read_csv(file)

    df['start_frame'] = df['start'].apply(lambda x: time_to_frame(x))

    new_rows = []

    counter = 0

    for _, row in df.iterrows():

        start_frame = row['start_frame']

        if start_frame == counter:
            new_row = row.copy()
            new_row['frame'] = start_frame
            new_rows.append(new_row)
            counter += 1

    new_df = pandas.DataFrame(new_rows)
    new_df.drop(columns=["start_frame"])

    new_df.to_csv(f"preprocessing/final_features/audio/{participant}_audio.csv", index=False)


directory = "preprocessing/nodbot_filtered_audio_features"

for filename in os.listdir(directory):

    url = os.path.join(directory, filename)

    participant = url.split("\\")[-1].split("_")[0]

    create_frames_audio_csv(url, participant)
    print(participant)