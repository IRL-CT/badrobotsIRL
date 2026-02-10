"""
Combine all CSVs in nodbot_audio_features into a single DataFrame.

- Removes the 'file' column
- Converts 'start' to a 1-indexed 'frame' column
- Adds a 'participant' column derived from the filename (e.g. "p2nodbot")
- Saves the result to nodbot_audio_features_combined.csv
"""

import os
import pandas as pd

INPUT_DIR = "nodbot_audio_features"
OUTPUT_FILE = "opensmile.csv"

all_dfs = []

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith("_audio_features.csv"):
        continue

    filepath = os.path.join(INPUT_DIR, filename)

    # Extract participant name: e.g. "p2nodbot.wav_audio_features.csv" -> "p2nodbot"
    participant = filename.replace(".wav_audio_features.csv", "")

    if participant in ("p3nodbot", "p24nodbot"):  # excluded participants
        continue

    df = pd.read_csv(filepath)

    # Drop the 'file' column
    df = df.drop(columns=["file"])

    # Replace 'start' with 'frame' (1-indexed integer)
    df = df.drop(columns=["start"])
    df.insert(0, "audio_frame", range(1, len(df) + 1))

    df = df.drop(columns=["end"])

    # Add participant column
    df.insert(1, "participant", participant)

    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(OUTPUT_FILE, index=False)

print(f"Combined {len(all_dfs)} files → {len(combined)} rows")
print(f"Columns: {list(combined.columns)}")
print(f"Saved to {OUTPUT_FILE}")
print(combined.head())
