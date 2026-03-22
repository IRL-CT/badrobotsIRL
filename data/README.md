# Data Directory

This directory contains all CSV datasets used by the training scripts. Need to generate or copy them into the appropriate subdirectories.

## Directory Structure

| Directory | Contents
|---|---
| `raw/` | Pre-interpolation datasets (`all_participants_0_3.csv`, merged features)
| `interpolated/` | 100fps interpolated datasets (`allparticipants_100fps.csv`, `curated_features_v5_100fps.csv`, etc.)
| `feature_sets/` | Derived feature sets (rf, selectkbest, curated, catch22, tsfresh)
| `embeddings/` | CLIP text embeddings + cosine similarity
| `individual/` | Per-modality feature CSVs (audio, facial, pose with norm/pca variants)
| `pose/` | Per-participant pose keypoint CSVs

## Key Files

### `interpolated/`
- `allparticipants_100fps.csv` — full feature set (92 features) at 100fps
- `curated_features_v5_100fps.csv` — curated feature set (31 features) at 100fps

### `feature_sets/`
- `rf_allparticipants_100fps.csv` — random forest selected features
- `select_k_best_allparticipants_100fps_binary_label.csv` — SelectKBest for binary
- `select_k_best_allparticipants_100fps_multiclass_label.csv` — SelectKBest for multiclass

### `embeddings/`
- `clip_text_embeddings_pca.csv` — PCA-reduced CLIP text embeddings
- `clip_text_cosine_similarity.csv` — CLIP cosine similarity features
