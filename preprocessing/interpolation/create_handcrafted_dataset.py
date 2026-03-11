import pandas as pd

# Input file containing all features (including the newly computed zones)
INPUT_PATH = 'curated_features_dataset_v5_zones.csv'
# Output file for just the handcrafted features dataset
OUTPUT_PATH = 'handcrafted_allparticipants_100fps.csv'

print(f"Loading '{INPUT_PATH}'...")
df = pd.read_csv(INPUT_PATH)

# Define columns to keep
# We need the metadata/labels plus only the handcrafted features
cols_to_keep = [
    'frame', 
    'participant', 
    'binary_label', 
    'multiclass_label',
    'zone_head_heading', 
    'zone_head_pitch', 
    'zone_trunk_tilt', 
    'zone_distance', 
    'zone_facial_expression', 
]

print(f"Subsetting data with columns: {cols_to_keep}")
handcrafted_df = df[cols_to_keep]

# Save the pure handcrafted dataset
handcrafted_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH}")
print(f"Shape: {handcrafted_df.shape}")
print(handcrafted_df.head())
