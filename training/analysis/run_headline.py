#!/usr/bin/env python3
"""Re-run the 7 headline configs, saving per-prediction probabilities (P0-7).

Why a separate file instead of patching runs_registry.csv: the registries are
already 3.8 GB and adding y_proba would multiply that. Only these 7 runs need
per-prediction analysis, so they get their own registry with exactly the schema
`analysis/registry_to_seconds.load_run` expects:

    run_id, fold, y_pred, y_true, y_proba, session, frame_idx   (JSON lists)

How it works: rather than reimplementing the training pipeline (and risking it
drifting from the real one), this imports the actual model module and runs its
`train()` with a stubbed wandb whose config is the fixed headline config. Two
interception points capture what the registry needs:

  * `apply_windowing` -> the windowed frame and its `last_positions`, which give
    the original 100 fps row index behind every windowed row.
  * `get_all_metrics` -> called once per fold with exactly (y_pred, y_true,
    y_pred_proba, sessions) for the test split.

frame_idx is reconstructed by walking each test participant's windowed rows in
order, which is valid because `create_data_splits` selects test rows with a
boolean mask and so preserves within-participant ordering.

Usage
-----
    cd training/analysis
    python3 run_headline.py                 # all 7
    python3 run_headline.py --only 2 5      # a subset
    python3 run_headline.py --list          # show the configs and exit
"""

import argparse
import importlib
import json
import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.normpath(os.path.join(HERE, "..", "temp_allscripts"))
V5 = os.path.normpath(os.path.join(HERE, "..", "recovery"))
COMMON = os.path.normpath(os.path.join(HERE, "..", "common"))

OUT = os.path.join(HERE, "headline_predictions.csv")

# Best config per (track, task, input family), taken from the W&B exports:
#   wandb_detection_linear.csv / wandb_recovery_linear.csv, max avg_test_f1.
# `track` picks which folder's scripts run: v4 = detection (recovery block
# dropped), v5 = recovery (ERROR vs RECOVERY / NEUTRAL-ERROR-RECOVERY).
HEADLINE = {
    1: dict(track="v4", name="detect_bin_engineered", model="rf",
            binary_multiclass="binary", modality="audio_text", feature_set="rf",
            dataset="norm", window=15, stride=3, aggregation="average",
            gemini_source="visual_audio", gemini_embedding="256d"),
    2: dict(track="v4", name="detect_bin_embed_incl", model="rf",
            binary_multiclass="binary", modality="pose_facial_text_gemini",
            feature_set="full", dataset="norm", window=10, stride=3,
            aggregation="last", gemini_source="visual_audio", gemini_embedding="full"),
    3: dict(track="v4", name="detect_bin_embed_only", model="sgd",
            binary_multiclass="binary", modality="gemini", feature_set="selectkbest",
            dataset="pca", window=15, stride=1, aggregation="last",
            gemini_source="visual_audio", gemini_embedding="256d"),
    4: dict(track="v5", name="recov_bin_engineered", model="rf",
            binary_multiclass="binary", modality="pose_facial_audio_cosine",
            feature_set="full", dataset="norm", window=10, stride=10,
            aggregation="last", gemini_source="visual", gemini_embedding="256d"),
    5: dict(track="v5", name="recov_bin_embed_incl", model="rf",
            binary_multiclass="binary", modality="pose_facial_audio_gemini",
            feature_set="full", dataset="norm", window=10, stride=10,
            aggregation="average", gemini_source="visual_audio", gemini_embedding="128d"),
    6: dict(track="v5", name="recov_3cls_engineered", model="rf",
            binary_multiclass="multiclass", modality="facial_audio",
            feature_set="full", dataset="pca", window=30, stride=10,
            aggregation="average", gemini_source="visual_audio", gemini_embedding="128d"),
    7: dict(track="v5", name="recov_3cls_embed_incl", model="svm",
            binary_multiclass="multiclass", modality="text_gemini",
            feature_set="selectkbest", dataset="norm", window=15, stride=5,
            aggregation="mode", gemini_source="visual_audio", gemini_embedding="full"),
}

# Hyperparameters the sweeps also set. Fixed at the grid's mid value: these runs
# exist to produce PREDICTIONS for the time-matched / per-participant analyses,
# not to re-find the optimum.
MODEL_DEFAULTS = {
    "rf":  dict(n_estimators=300, max_depth=15),
    "sgd": dict(alpha=1e-4, max_iter=1000, tol=1e-3),
    "svm": dict(C=1, kernel="rbf", gamma="scale"),
    "mlp": dict(mlp_hidden_layer_sizes=(100,), mlp_activation="relu"),
    "knn": dict(n_neighbors=7),
}
COMMON_DEFAULTS = dict(test_window_size=5, test_stride=5, agg_features=False,
                       feature_randomizer=1, model_type=None)


class Cfg(dict):
    """Stands in for wandb.config: attribute access AND dict()/`in`."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def build_cfg(spec):
    c = dict(COMMON_DEFAULTS)
    c.update(MODEL_DEFAULTS[spec["model"]])
    c.update({k: v for k, v in spec.items() if k not in ("track", "name", "model")})
    c["model_type"] = spec["model"]
    c["class"] = spec["binary_multiclass"]      # deep scripts read config["class"]
    return Cfg(c)


def run_one(idx, spec):
    folder = V4 if spec["track"] == "v4" else V5
    for p in (folder, COMMON):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, COMMON)
    sys.path.insert(0, folder)
    for m in list(sys.modules):
        if m.split(".")[0] in {"mlp_model", "rf_model", "sgd_model", "svm_model",
                               "knn_model", "eval_utils", "label_utils",
                               "recovery_labels", "feature_sets", "registry_utils",
                               "create_data_splits", "get_all_metrics"}:
            del sys.modules[m]

    captured = {"win": None, "folds": []}

    stub = types.ModuleType("wandb")
    stub.config = build_cfg(spec)
    stub.init = lambda *a, **k: None
    stub.log = lambda d: None
    stub.run = types.SimpleNamespace(id=f"headline_{idx}_{spec['name']}")
    sys.modules["wandb"] = stub

    mod = importlib.import_module(f"{spec['model']}_model")

    # -- interception 1: windowed frame + last_positions -----------------------
    _orig_win = mod.apply_windowing

    def win(*a, **k):
        df, lp = _orig_win(*a, **k)
        captured["win"] = (df, list(lp) if lp is not None else None)
        return df, lp
    mod.apply_windowing = win

    # -- interception 2: per-fold test predictions -----------------------------
    _orig_metrics = mod.get_all_metrics

    def metrics(y_pred, y_true, y_pred_proba=None, sessions=None, **k):
        captured["folds"].append(dict(
            y_pred=np.asarray(y_pred), y_true=np.asarray(y_true),
            y_proba=None if y_pred_proba is None else np.asarray(y_pred_proba, dtype=float),
            sessions=None if sessions is None else np.asarray(sessions)))
        return _orig_metrics(y_pred, y_true, y_pred_proba=y_pred_proba,
                             sessions=sessions, **k)
    mod.get_all_metrics = metrics

    # The scripts do `from registry_utils import append_to_registry`, so the name
    # lives on the model module, not as a `registry_utils` attribute. Patch both
    # the module-level name and the source module, and don't assume either.
    if hasattr(mod, "append_to_registry"):
        mod.append_to_registry = lambda *a, **k: None
    _ru = sys.modules.get("registry_utils")
    if _ru is not None:
        _ru.append_to_registry = lambda *a, **k: None

    print(f"\n=== [{idx}] {spec['name']} ({spec['track']}, {spec['model']}, "
          f"{spec['binary_multiclass']}, {spec['modality']}) ===", flush=True)
    mod.train()

    # -- reconstruct frame_idx and emit registry rows --------------------------
    if captured["win"] is None:
        raise RuntimeError("apply_windowing never fired; cannot recover frame_idx")
    wdf, last_positions = captured["win"]
    part_col = wdf.columns[1]
    win_parts = wdf[part_col].to_numpy()

    # per participant, the original 100 fps row behind each windowed row, in order
    pos_by_part = {}
    for p in pd.unique(win_parts):
        pos_by_part[p] = np.asarray(last_positions)[win_parts == p]

    rows = []
    for fold, f in enumerate(captured["folds"]):
        sess = f["sessions"]
        if sess is None:
            raise RuntimeError("sessions missing; cannot align frame_idx")
        cursor = {p: 0 for p in pos_by_part}
        frame_idx = np.empty(len(sess), dtype=int)
        for i, p in enumerate(sess):
            frame_idx[i] = pos_by_part[p][cursor[p]]
            cursor[p] += 1
        pr = f["y_proba"]
        if pr is not None and pr.ndim == 1:
            pr = pr[:, None]
        rows.append(dict(
            run_id=stub.run.id, run_name=spec["name"], track=spec["track"],
            label_scheme=spec["binary_multiclass"], fold=fold,
            y_pred=json.dumps(f["y_pred"].tolist()),
            y_true=json.dumps(f["y_true"].tolist()),
            y_proba=json.dumps([] if pr is None else pr.tolist()),
            session=json.dumps(sess.tolist()),
            frame_idx=json.dumps(frame_idx.tolist())))
        print(f"  fold {fold}: {len(sess)} rows, "
              f"{len(np.unique(sess))} participants", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for i, s in HEADLINE.items():
            print(f"{i}: {s['name']:24s} {s['track']} {s['model']:4s} "
                  f"{s['binary_multiclass']:10s} {s['modality']}")
        return 0

    todo = a.only or sorted(HEADLINE)
    # Resume: skip configs already written. The first attempt ran as a login-node
    # nohup, died with the session, and wrote nothing because output was held to
    # the end. Append after every config instead.
    done = set()
    if os.path.exists(a.out):
        done = set(pd.read_csv(a.out, usecols=["run_name"]).run_name.unique())
        print(f"resuming; already have: {sorted(done)}", flush=True)

    all_rows, failed = [], []
    for i in todo:
        if HEADLINE[i]["name"] in done:
            print(f"  [{i}] {HEADLINE[i]['name']}: already done, skipping", flush=True)
            continue
        try:
            rows = run_one(i, HEADLINE[i])
            pd.DataFrame(rows).to_csv(a.out, mode="a",
                                      header=not os.path.exists(a.out), index=False)
            all_rows += rows
            print(f"  [{i}] appended {len(rows)} fold-rows -> {a.out}", flush=True)
        except Exception as e:
            print(f"  [{i}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append((i, f"{type(e).__name__}: {e}"))

    if all_rows:
        print(f"\nwrote {len(all_rows)} new fold-rows -> {a.out}")
    print(f"succeeded: {len(todo) - len(failed)}/{len(todo)}")
    for i, e in failed:
        print(f"  FAILED {i}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
