# Multimodal Embedding Pipeline (`preprocessing/embeddings/`)

This directory contains all generation, combination, normalization, and dimensionality reduction scripts for embeddings used in BadRobotsIRL.

---

## Important Architectural Distinction: Visual-Only vs. Audiovisual

### Google Gemini Embedding 2 Nuance
Google's official documentation for `gemini-embedding-2` specifies:
> **"Audio tracks are not processed in video files."**

When an `.mp4` container is submitted to `client.models.embed_content()`, the API extracts only visual frames and **discards the audio stream**. Therefore:
1. **Visual-Only**: Uses `.mp4` video clips (audio stripped via `-an`). The model samples up to 32 frames across the clip.
2. **Audiovisual (Multimodal)**: Video and audio must be submitted as separate, interleaved `Part` objects within the single API call:
   ```python
   contents = [
       types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
       types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
   ]
   ```
   Both streams are jointly attended to, producing a unified 3,072-dimensional multimodal embedding.

---

## Scripts Overview

| Script | Purpose | Checkpoint / Output |
| :--- | :--- | :--- |
| `generate_gemini_visual_only_embeddings.py` | Generates 1-s visual-only embeddings from 480p MP4 clips. | `data/embeddings/gemini_checkpoints_visual_only/{p}_gemini_visual.csv` |
| `generate_gemini_audiovisual_embeddings.py` | Generates 1-s multimodal embeddings from interleaved 480p MP4 + 16kHz WAV clips. | `data/embeddings/gemini_checkpoints_audiovisual/{p}_gemini_audiovisual.csv` |
| `combine_gemini_embeddings.py` | Combines per-participant checkpoints into unified datasets with MRL truncation (768D, 256D, 128D) and L2-normalization. | `data/embeddings/gemini_video_embeddings_{mode}_{full,768d,256d,128d}.csv` |
| `do_pca_gemini_embeddings.py` | Fits PCA (default 90% variance) and transforms the full dataset. | `data/embeddings/gemini_video_embeddings_pca_{mode}.csv` |
| `do_pca_text_embeddings.py` | Fits PCA and calculates cosine similarity on text embeddings. | `data/embeddings/clip_text_embeddings_pca.csv` |

---

## Parity Guarantee with Base Dataset

In all training pipelines (`training/linear_classifiers/`, `training/rnn/`, `training/transformer/`), feature sets are aligned via direct row indexing:
```python
df_gemini_index = df_gemini_raw.iloc[last_positions, 4:].reset_index(drop=True)
```
Consequently:
1. **Row Count Parity**: Each checkpoint row count is locked to `len(df_base[df_base['participant'] == p])`.
2. **Participant Ordering**: Checkpoints are concatenated in the **exact sequential order of first appearance** in `data/interpolated/allparticipants_100fps.csv` (`p10nodbot`, `p11nodbot`, ..., `p9nodbot`).
3. **Automated Assertion Gate**: `combine_gemini_embeddings.py` automatically asserts that total rows, participant sequence, frame numbers, and binary/multiclass labels match `allparticipants_100fps.csv` 100% identically before saving.

---

## Usage Workflow

### 1. Visual-Only Pipeline
* **Generate checkpoints**:
  ```bash
  python3 generate_gemini_visual_only_embeddings.py
  ```
* **Combine into unified & MRL datasets**:
  ```bash
  python3 combine_gemini_embeddings.py --mode visual_only
  ```
* **Run PCA**:
  ```bash
  python3 do_pca_gemini_embeddings.py --mode visual_only
  ```

### 2. Audiovisual Pipeline (Run when quota resets)
* **Generate checkpoints**:
  ```bash
  python3 generate_gemini_audiovisual_embeddings.py
  ```
* **Combine into unified & MRL datasets**:
  ```bash
  python3 combine_gemini_embeddings.py --mode audiovisual
  ```
* **Run PCA**:
  ```bash
  python3 do_pca_gemini_embeddings.py --mode audiovisual
  ```
