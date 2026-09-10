"""
Shared registry utilities for tracking completed model runs.

All model scripts (knn_model.py, mlp_model.py, rf_model.py, etc.) should call
`append_to_registry()` at the end of each training run. Predictions for every
fold are serialised as JSON lists and stored directly in the CSV — one row per
(run_id, fold) — so no wandb artifacts are required.

Registry columns
----------------
run_id              wandb run ID (short hash)
model_type          'knn', 'mlp', 'rf', 'svm', 'sgd', …
binary_multiclass   'binary' or 'multiclass'
feature_set         'full', 'rf', 'catch22', 'tsfresh', 'curated_v1', 'curated_v3'
window              dataset-level aggregation window size (int)
stride              dataset-level aggregation stride (int)
aggregation         'average', 'mode', 'last', or 'N/A' for catch22/tsfresh
dataset             'reg', 'norm', or 'pca'
modality            e.g. 'pose_facial_audio', or 'all' for no-modality-selection sets
test_window_size    window size used for test windowed metrics
test_stride         stride used for test windowed metrics
fold                fold index (0–4)
y_pred              JSON list of per-sample predictions for this fold
y_true              JSON list of ground-truth labels for this fold
session             JSON list of session IDs for this fold
timestamp           ISO-8601 UTC timestamp of the run
"""

import csv
import json
import os
from datetime import datetime, timezone

import numpy as np

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "runs_registry.csv")

REGISTRY_COLS = [
    "run_id",
    "model_type",
    "binary_multiclass",
    "feature_set",
    "window",
    "stride",
    "aggregation",
    "dataset",
    "modality",
    "test_window_size",
    "test_stride",
    "fold",
    "y_pred",
    "y_true",
    "session",
    "timestamp",
]


def _to_json_list(arr) -> str:
    """Convert a numpy array or list to a compact JSON string."""
    return json.dumps(arr.tolist() if isinstance(arr, np.ndarray) else list(arr))


def append_to_registry(
    run_id: str,
    config: dict,
    all_fold_predictions: "list[pd.DataFrame]",
) -> None:
    """Append one row per fold to the registry CSV.

    Parameters
    ----------
    run_id : str
        The wandb run ID (``wandb.run.id``).
    config : dict
        The run's wandb config (``dict(wandb.config)``).
    all_fold_predictions : list of pd.DataFrame
        One DataFrame per fold, each with columns:
        fold, y_pred, y_true, session.
    """
    write_header = not os.path.exists(REGISTRY_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(REGISTRY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        if write_header:
            writer.writeheader()

        for fold_df in all_fold_predictions:
            fold_idx = int(fold_df["fold"].iloc[0])
            row = {
                "run_id": run_id,
                "model_type": config.get("model_type", "unknown"),
                "binary_multiclass": config.get("binary_multiclass", ""),
                "feature_set": config.get("feature_set", ""),
                "window": config.get("window", ""),
                "stride": config.get("stride", ""),
                "aggregation": config.get("aggregation", "N/A"),
                "dataset": config.get("dataset", ""),
                "modality": config.get("modality", "all"),
                "test_window_size": config.get("test_window_size", ""),
                "test_stride": config.get("test_stride", ""),
                "fold": fold_idx,
                "y_pred": _to_json_list(fold_df["y_pred"].values),
                "y_true": _to_json_list(fold_df["y_true"].values),
                "session": _to_json_list(fold_df["session"].values),
                "timestamp": timestamp,
            }
            writer.writerow(row)

    print(f"[registry] Logged run {run_id} ({config.get('model_type', '?')}, "
          f"{len(all_fold_predictions)} folds) → {REGISTRY_PATH}")


def load_registry() -> "pd.DataFrame":
    """Load the registry CSV.

    Returns an empty DataFrame with the correct columns if the file does not
    exist yet.
    """
    import pandas as pd

    if not os.path.exists(REGISTRY_PATH):
        return pd.DataFrame(columns=REGISTRY_COLS)

    df = pd.read_csv(REGISTRY_PATH, dtype=str)

    # Cast numeric metadata columns
    for col in ("window", "stride", "test_window_size", "test_stride", "fold"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_predictions_for_run(registry: "pd.DataFrame", run_id: str) -> "dict[int, dict]":
    """Return a dict mapping fold index → {y_pred, y_true, session} arrays
    for a given run_id.
    """
    run_rows = registry[registry["run_id"] == run_id]
    result = {}
    for _, row in run_rows.iterrows():
        fold = int(row["fold"])
        result[fold] = {
            "y_pred": np.array(json.loads(row["y_pred"])),
            "y_true": np.array(json.loads(row["y_true"])),
            "session": np.array(json.loads(row["session"])),
        }
    return result


def filter_compatible_runs(
    registry: "pd.DataFrame",
    binary_multiclass: str,
    window: int,
    stride: int,
    aggregation: str = "any",
    model_type_filter: str = "mixed",
) -> "pd.DataFrame":
    """Return unique run_ids whose metadata is compatible with the given config.

    Operates on the full (per-fold) registry but deduplicates on run_id so
    callers get one representative row per run for sampling.

    Parameters
    ----------
    aggregation : str
        'any' → do not filter by aggregation (mixes catch22/tsfresh with others).
        Otherwise must match exactly.
    model_type_filter : str
        'mixed' → accept all model types.
        Any specific value ('knn', 'mlp', …) → restrict to that type only.
    """
    mask = (
        (registry["binary_multiclass"] == binary_multiclass)
        & (registry["window"] == window)
        & (registry["stride"] == stride)
    )

    if aggregation != "any":
        mask &= (registry["aggregation"] == aggregation) | registry["aggregation"].isna()

    if model_type_filter != "mixed":
        mask &= registry["model_type"] == model_type_filter

    compatible = registry[mask].copy()

    # Return one row per run_id (use first fold row as the metadata representative)
    return compatible.drop_duplicates(subset="run_id", keep="first")
