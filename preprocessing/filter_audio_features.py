import csv
import pandas
from datetime import datetime

def get_participant(file):
    with open(file, newline='') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        second_row = next(csvreader)
        return second_row[0].split("/")[-1].split(".")[0]

def read_speaking_file(file):
    with open(file, newline="") as f:
        return list(csv.DictReader(f))

def write_processed_features(file, fieldnames, rows):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

def speaking_range(file):
    for row in file:
        start = row['start']
        end = row['end']
    return start, end


for i in range(25,30):
    speaking_file = f"./speaker_diarization_csv/p{i}nodbot_audio.csv"
    features_file = f"./nodbot_audio_features/p{i}nodbot.wav_audio_features.csv"

    speaking = read_speaking_file(speaking_file)
    features = read_speaking_file(features_file)
    participant_num = get_participant(features_file)

    filtered_features = []

    for feature_row in features:
        start = feature_row['start'].split(":")[-1]
        end = feature_row['end'].split(":")[-1]
        matched = False
        for speaking_row in speaking:
            start_speaking = speaking_row['start']
            end_speaking = speaking_row['end']
            if speaking_row['speaker'] == "participant" and (start_speaking <= start and end <= end_speaking):
                filtered_features.append(feature_row)
                matched = True
                break
        if not matched:
            empty_row = {key: feature_row[key] if key in ['start', 'end'] else '' for key in feature_row.keys()}
            filtered_features.append(empty_row)
        

    filtered_features_values = []

    for j in range(len(filtered_features)):
        filtered_features_values.append(filtered_features[j].values())
        
    #print(filtered_features_values)

    write_processed_features(f'./nodbot_filtered_audio_features/{participant_num}_processed_features.csv', filtered_features[0].keys(), filtered_features_values)

