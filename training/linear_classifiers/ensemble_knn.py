"""
Manual ensemble of specific wandb runs via majority vote.

Predictions are loaded from the local runs_registry.csv — no wandb artifact
downloads required.

Usage
-----
python ensemble_knn.py \\
    --project brirl_linear_inter \\
    --runs <run_id_1> <run_id_2> <run_id_3> \\
    [--test_window_size 10] \\
    [--test_stride 5] \\
    [--entity <wandb_entity>]

The ensemble result is logged as a new wandb run in the same project.

Notes
-----
- For the majority vote to be valid, all selected runs must share the same
  binary_multiclass, window, stride, and aggregation config so that their
  test splits contain the same samples in the same order.
- The script will warn (and exclude the offending run per fold) if y_true
  arrays differ, but it is the user's responsibility to select compatible runs.
"""

import argparse

import numpy as np
import wandb
from scipy import stats
from sklearn.metrics import confusion_matrix

from get_all_metrics import get_all_metrics
from registry_utils import get_predictions_for_run, load_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    """Per-sample majority vote across a list of prediction arrays."""
    stacked = np.stack(predictions, axis=0)
    voted = np.apply_along_axis(
        lambda col: stats.mode(col, keepdims=False).mode,
        axis=0,
        arr=stacked,
    )
    return voted


# ---------------------------------------------------------------------------
# Main ensemble logic
# ---------------------------------------------------------------------------

def ensemble_runs(
    run_ids: list[str],
    project: str,
    entity: str | None = None,
    test_window_size: int | None = None,
    test_stride: int | None = None,
) -> None:
    """Load predictions from the registry for the given run IDs and compute
    a majority-vote ensemble. Logs results as a new wandb run.
    """
    registry = load_registry()

    if registry.empty:
        raise RuntimeError(
            "runs_registry.csv is empty or does not exist. "
            "Run some models first with knn_model.py."
        )

    # ------------------------------------------------------------------ #
    # Load predictions from registry                                       #
    # ------------------------------------------------------------------ #
    run_preds = {}
    missing = []
    for run_id in run_ids:
        preds = get_predictions_for_run(registry, run_id)
        if preds:
            run_preds[run_id] = preds
            print(f"  ✓ Loaded predictions for run {run_id}")
        else:
            missing.append(run_id)
            print(f"  ✗ Run {run_id} not found in registry — skipping.")

    if not run_preds:
        raise RuntimeError("None of the specified run IDs were found in the registry.")

    # ------------------------------------------------------------------ #
    # Metadata summary from registry                                       #
    # ------------------------------------------------------------------ #
    run_meta = registry.drop_duplicates(subset="run_id").set_index("run_id")
    model_types = {rid: run_meta.loc[rid, "model_type"] if rid in run_meta.index else "?"
                   for rid in run_preds}

    # ------------------------------------------------------------------ #
    # Initialise wandb run                                                 #
    # ------------------------------------------------------------------ #
    wandb.init(
        project=project,
        entity=entity,
        job_type="ensemble",
        config={
            "run_ids": list(run_preds.keys()),
            "missing_run_ids": missing,
            "n_models": len(run_preds),
            "aggregation": "majority_vote",
            "model_types": model_types,
            "test_window_size": test_window_size,
            "test_stride": test_stride,
        },
    )

    # ------------------------------------------------------------------ #
    # Per-fold ensemble                                                    #
    # ------------------------------------------------------------------ #
    folds = sorted(next(iter(run_preds.values())).keys())

    test_metrics_list: dict[str, list] = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": [],
        "test_auc": [],
        "test_fnr": [],
        "test_windowed_accuracy": [],
        "test_windowed_precision": [],
        "test_windowed_recall": [],
        "test_windowed_f1": [],
        "test_earliest_detection_time": [],
    }

    for fold in folds:
        preds_this_fold: list[np.ndarray] = []
        y_true_ref = None
        sessions_ref = None
        runs_in_fold = 0

        for run_id, fold_data in run_preds.items():
            if fold not in fold_data:
                print(f"  Fold {fold}: run {run_id} has no data — skipping.")
                continue

            data = fold_data[fold]

            if y_true_ref is None:
                y_true_ref = data["y_true"]
                sessions_ref = data["session"]
            else:
                if not np.array_equal(y_true_ref, data["y_true"]):
                    print(
                        f"  Fold {fold}: y_true mismatch for run {run_id} "
                        "(incompatible test split) — excluding from this fold."
                    )
                    continue

            preds_this_fold.append(data["y_pred"])
            runs_in_fold += 1

        wandb.log({f"t{fold}_n_models_in_fold": runs_in_fold})

        if not preds_this_fold:
            print(f"  Fold {fold}: no valid predictions — skipping.")
            continue

        y_pred_ensemble = majority_vote(preds_this_fold)

        test_metrics = get_all_metrics(
            y_pred_ensemble,
            y_true_ref,
            y_pred_proba=None,
            sessions=sessions_ref,
            window_size=test_window_size,
            stride=test_stride,
            tolerance=1,
        )
        test_metrics.pop("test_edt_per_session", None)

        for key in test_metrics:
            if key in test_metrics_list:
                test_metrics_list[key].append(test_metrics[key])

        wandb.log({f"t{fold}_{k}": v for k, v in test_metrics.items()})

        cm = confusion_matrix(y_true_ref, y_pred_ensemble)
        print(f"  Fold {fold} ({runs_in_fold} models): {test_metrics}")
        print(f"  Confusion matrix:\n{cm}")

    # ------------------------------------------------------------------ #
    # Average metrics                                                      #
    # ------------------------------------------------------------------ #
    avg_metrics = {
        f"avg_{k}": np.mean(v)
        for k, v in test_metrics_list.items()
        if v
    }
    wandb.log(avg_metrics)
    wandb.finish()

    print("\nEnsemble Average Metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Majority-vote ensemble of specific wandb runs (reads from registry).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--project", required=True,
                        help="wandb project name (e.g. brirl_linear_inter)")
    parser.add_argument("--runs", required=True, nargs="+",
                        help="Two or more wandb run IDs to ensemble")
    parser.add_argument("--entity", default=None,
                        help="wandb entity. Defaults to your wandb default.")
    parser.add_argument("--test_window_size", type=int, default=None)
    parser.add_argument("--test_stride", type=int, default=None)
    args = parser.parse_args()

    if len(args.runs) < 2:
        parser.error("Provide at least 2 run IDs to ensemble.")

    print(f"Ensembling {len(args.runs)} runs from project '{args.project}':")
    for r in args.runs:
        print(f"  - {r}")

    ensemble_runs(
        run_ids=args.runs,
        project=args.project,
        entity=args.entity,
        test_window_size=args.test_window_size,
        test_stride=args.test_stride,
    )


if __name__ == "__main__":
    main()
