import pandas as pd
import os

directory = "preprocessing/merged_features"

csv_files = []

for filename in os.listdir(directory):
    url = os.path.join(directory, filename)
    csv_files.append(url)

print(csv_files)

dfs = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(dfs, ignore_index=True)

print(combined_df)

combined_df.to_csv("preprocessing/merged_features/all_participants.csv", index=False)