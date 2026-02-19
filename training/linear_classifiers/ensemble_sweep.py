"""
Wandb sweep over ensemble configurations.

All predictions are read directly from the local runs_registry.csv — no
wandb artifact downloads required. Each row in the registry corresponds to
one (run_id, fold) pair and contains the per-sample y_pred / y_true / session
arrays as JSON lists.

For each sweep run the script will:
  1. Load runs_registry.csv
  2. Filter for runs compatible with this sweep's binary_multiclass / window /
     stride / aggregation / model_type_filter settings
  3. Randomly sample up to n_models runs (uses all available if fewer exist)
  4. Per fold: majority-vote across selected models
  5. Compute all standard metrics and log everything to wandb

Usage
-----
python ensemble_sweep.py --project brirl_linear_inter [--entity <entity>] [--count N]
"""

import numpy as np
import wandb
from scipy import stats
from sklearn.metrics import confusion_matrix

from get_all_metrics import get_all_metrics
from registry_utils import filter_compatible_runs, get_predictions_for_run, load_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    """Per-sample majority vote across a list of model prediction arrays."""
    stacked = np.stack(predictions, axis=0)          # (n_models, n_samples)
    voted = np.apply_along_axis(
        lambda col: stats.mode(col, keepdims=False).mode,
        axis=0,
        arr=stacked,
    )
    return voted


# ---------------------------------------------------------------------------
# One ensemble run (called by the wandb agent)
# ---------------------------------------------------------------------------

def ensemble_run():
    wandb.init()
    config = wandb.config
    print(f"\n{'='*60}")
    print(f"Ensemble config: {dict(config)}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 1. Load registry and filter compatible runs                          #
    # ------------------------------------------------------------------ #
    registry = load_registry()

    if registry.empty:
        print("Registry is empty — no runs available. Skipping.")
        wandb.log({"status": "registry_empty", "n_models_available": 0, "n_models_used": 0})
        return

    compatible_runs = filter_compatible_runs(
        registry,
        binary_multiclass=config.binary_multiclass,
        window=config.window,
        stride=config.stride,
        aggregation=config.aggregation,
        model_type_filter=config.model_type_filter,
    )

    n_available = len(compatible_runs)
    n_requested = config.n_models
    n_actual = min(n_available, n_requested)

    wandb.log({
        "n_models_requested": n_requested,
        "n_models_available": n_available,
        "n_models_used": n_actual,
    })

    if n_actual == 0:
        print("No compatible runs found for this config. Skipping.")
        wandb.log({"status": "no_compatible_runs"})
        return

    # ------------------------------------------------------------------ #
    # 2. Sample runs                                                       #
    # ------------------------------------------------------------------ #
    rng_seed = abs(hash(wandb.run.id)) % (2**31)
    selected = compatible_runs.sample(n=n_actual, random_state=rng_seed)

    selected_run_ids = selected["run_id"].tolist()
    selected_model_types = selected["model_type"].tolist()

    wandb.log({
        "selected_run_ids": str(selected_run_ids),
        "selected_model_types": str(selected_model_types),
        "selected_model_type_counts": str({
            mt: selected_model_types.count(mt) for mt in set(selected_model_types)
        }),
    })
    print(f"Selected {n_actual} runs:")
    for rid, mt in zip(selected_run_ids, selected_model_types):
        print(f"  {rid}  ({mt})")

    # ------------------------------------------------------------------ #
    # 3. Load predictions from registry (no network required)              #
    # ------------------------------------------------------------------ #
    # run_preds[run_id][fold] = {y_pred, y_true, session}
    run_preds = {}
    for run_id in selected_run_ids:
        preds = get_predictions_for_run(registry, run_id)
        if preds:
            run_preds[run_id] = preds
        else:
            print(f"  No predictions found in registry for run {run_id} — skipping.")

    wandb.log({"n_models_loaded": len(run_preds)})

    if not run_preds:
        print("No predictions could be loaded. Skipping.")
        wandb.log({"status": "load_failed"})
        return

    # ------------------------------------------------------------------ #
    # 4. Per-fold majority-vote ensemble                                   #
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
                print(f"  Fold {fold}: run {run_id} missing — skipping this run.")
                continue

            data = fold_data[fold]

            if y_true_ref is None:
                y_true_ref = data["y_true"]
                sessions_ref = data["session"]
            else:
                if not np.array_equal(y_true_ref, data["y_true"]):
                    print(
                        f"  Fold {fold}: y_true mismatch for run {run_id} "
                        "(incompatible test split) — excluding."
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
            window_size=config.test_window_size,
            stride=config.test_stride,
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
    # 5. Average metrics across folds                                      #
    # ------------------------------------------------------------------ #
    avg_metrics = {
        f"avg_{k}": np.mean(v)
        for k, v in test_metrics_list.items()
        if v
    }
    wandb.log(avg_metrics)
    wandb.log({"status": "completed"})

    print("\nEnsemble Average Metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Sweep definition and entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch a wandb sweep over ensemble configurations.",
    )
    parser.add_argument("--project", required=True,
                        help="wandb project name (e.g. brirl_linear_inter_2).")
    parser.add_argument("--entity", default=None,
                        help="wandb entity. Defaults to your wandb default.")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of sweep runs to execute (default: unlimited).")
    args = parser.parse_args()

    sweep_config = {
        "method": "random",
        "name": "brirl_ensemble",
        "parameters": {
            # --------------------------------------------------------- #
            # Compatibility filters — runs selected from the registry     #
            # must match these so their test splits are aligned.          #
            # --------------------------------------------------------- #
            "binary_multiclass": {"values": ["binary", "multiclass"]},
            "window":            {"values": [1, 5, 10, 15, 30]},
            "stride":            {"values": [1, 3, 5, 10]},
            # 'any' = don't filter by aggregation
            "aggregation":       {"values": ["average", "mode", "last", "any"]},

            # --------------------------------------------------------- #
            # Ensemble-specific parameters                                #
            # --------------------------------------------------------- #
            "model_type_filter": {"values": ["knn", "mlp", "rf", "svm", "sgd", "mixed"]},
            "n_models":          {"values": [5, 10, 20, 50, 100]},

            # --------------------------------------------------------- #
            # Test-time windowed metric parameters                        #
            # --------------------------------------------------------- #
            "test_window_size":  {"values": [1, 5, 10, 15, 30]},
            "test_stride":       {"values": [1, 3, 5, 10]},
        },
    }

    print("Launching ensemble sweep:")
    print(sweep_config)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project,
        entity=args.entity,
    )
    wandb.agent(sweep_id, function=ensemble_run, count=args.count,
                entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
