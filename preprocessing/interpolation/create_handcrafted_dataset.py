import pandas as pd

# Input file containing all features (including zones + engineered features)
INPUT_PATH = 'curated_features_zones.csv'
# Output file for just the handcrafted features dataset
OUTPUT_PATH = 'curated_features_dataset_v5_100fps.csv'

print(f"Loading '{INPUT_PATH}'...")
df = pd.read_csv(INPUT_PATH)

# Define columns to keep
# Metadata/labels + zones + engineered audio deltas + cosine repetition metrics
cols_to_keep = [
    'frame', 
    'participant', 
    'binary_label', 
    'multiclass_label',
    # Zones
    'zone_head_heading', 
    'zone_head_pitch', 
    'zone_trunk_tilt', 
    'zone_facial_expression', 
    # Audio deltas (frame-to-frame)
    'Loudness_delta',
    'F0_delta',
    'jitter_delta',
    'shimmer_delta',
    'HNR_delta',
    # Cosine repetition metrics
    'high_repetition',
    'repetition_frequency',
]

print(f"Subsetting data with columns: {cols_to_keep}")
handcrafted_df = df[cols_to_keep]

# Save the pure handcrafted dataset
handcrafted_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH}")
print(f"Shape: {handcrafted_df.shape}")
print(handcrafted_df.head())
