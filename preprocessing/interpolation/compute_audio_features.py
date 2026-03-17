"""
Feature engineering on the curated features dataset (with zones).

Adds:
  1. Audio frame-to-frame deltas for 5 eGeMAPS descriptors
     (F0, Loudness, jitter, shimmer, HNR)
  2. Cosine repetition metrics derived from the Distance column
     (high_repetition flag, repetition_frequency rolling count)

Inputs:
  - curated_features_zones.csv             (v3 interpolated + zones)
  - opensmile_interpolated.csv             (source for jitter, shimmer, HNR)

Output:
  - curated_features_v5_100fps.csv
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_PATH = os.path.join(SCRIPT_DIR, 'curated_features_zones.csv')
OS_PATH = os.path.join(SCRIPT_DIR, 'opensmile_interpolated.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'curated_features_v5_100fps.csv')

# ── eGeMAPS descriptors for delta computation ──────────────────────────────────
# Columns already in v3:
V3_AUDIO_DELTA_COLS = {
    'Loudness_sma3':                'Loudness_delta',
    'F0semitoneFrom27.5Hz_sma3nz':  'F0_delta',
}
# Columns to pull from opensmile_interpolated:
OS_AUDIO_DELTA_COLS = {
    'jitterLocal_sma3nz':    'jitter_delta',
    'shimmerLocaldB_sma3nz': 'shimmer_delta',
    'HNRdBACF_sma3nz':      'HNR_delta',
}

# Cosine repetition threshold (similarity > 0.9 ↔ distance < 0.1)
HIGH_REP_THRESHOLD = 0.1
# Rolling window for repetition frequency (~0.5s at 100fps)
REP_WINDOW = 50

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading curated_features_zones.csv ...")
v3_df = pd.read_csv(V3_PATH)

print("Loading opensmile_interpolated.csv (for jitter, shimmer, HNR) ...")
os_df = pd.read_csv(OS_PATH)[['frame', 'participant'] + list(OS_AUDIO_DELTA_COLS.keys())]

# Merge the extra audio columns into v3
v3_df = pd.merge(v3_df, os_df, on=['frame', 'participant'], how='inner')
print(f"After merge: {v3_df.shape}")

# ── 1. Audio frame-to-frame deltas ────────────────────────────────────────────
all_delta_cols = {**V3_AUDIO_DELTA_COLS, **OS_AUDIO_DELTA_COLS}

for src_col, delta_col in all_delta_cols.items():
    v3_df[delta_col] = (
        v3_df.groupby('participant')[src_col]
        .transform(lambda s: s.diff().fillna(0))
    )

print(f"Added {len(all_delta_cols)} audio delta columns: {list(all_delta_cols.values())}")

# Drop raw jitter/shimmer/HNR — only deltas are needed
v3_df.drop(columns=list(OS_AUDIO_DELTA_COLS.keys()), inplace=True)
print(f"Dropped raw columns: {list(OS_AUDIO_DELTA_COLS.keys())}")

# ── 2. Cosine repetition metrics ──────────────────────────────────────────────
# Fill NaN Distance (e.g. first frames before any utterance) with 0
v3_df['Distance'] = v3_df['Distance'].fillna(0)

# high_repetition: binary flag where cosine distance < threshold
v3_df['high_repetition'] = (v3_df['Distance'] < HIGH_REP_THRESHOLD).astype(int)

# repetition_frequency: rolling sum of high_repetition in a window, per participant
v3_df['repetition_frequency'] = (
    v3_df.groupby('participant')['high_repetition']
    .transform(lambda s: s.rolling(REP_WINDOW, min_periods=1).sum())
)

print(f"Added cosine repetition features: high_repetition, repetition_frequency "
      f"(threshold={HIGH_REP_THRESHOLD}, window={REP_WINDOW})")

# ── Save ──────────────────────────────────────────────────────────────────────
v3_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {OUTPUT_PATH}")
print(f"Shape: {v3_df.shape}")
print(f"Columns: {list(v3_df.columns)}")
print(v3_df.head())
