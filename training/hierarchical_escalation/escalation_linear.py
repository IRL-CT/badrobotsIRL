#!/usr/bin/env python3
"""Escalation features + a shallow classifier (v6 linear arm).

Same hypothesis as hier_model.py -- that what transfers across people is
*change relative to that person's own baseline*, not absolute appearance -- but
tested with handcrafted features instead of a learned hierarchy. Cheap,
interpretable, and it runs in minutes rather than GPU-days, so it says quickly
whether the escalation idea is worth the recurrent machinery at all.

Per window, from the 1 Hz session:

    current        mean embedding over the window
    delta_base     current - mean of the participant's first `baseline_sec` s
    delta_prev     current - previous window
    drift          current - mean of all previous windows in the session
    step           L2 distance from the previous window (non-monotone)

The feature blocks are switchable so the sweep can isolate which part carries the
signal. `use_current=False, use_delta_base=True` is the pure "relative to own
baseline" model, which is the cleanest test of the hypothesis.

A caveat worth keeping in view: explicit per-participant baselining was already
tested on the raw per-second embeddings and made things slightly WORSE
(AUC 0.638 -> 0.615). These richer features may do better, but the prior is not
encouraging, and `use_current=True` alone is the control that reproduces the flat
model.

LEAKAGE -- read before adding any "time-like" feature
-----------------------------------------------------
The protocol is fixed: every session runs survey -> Error I -> II -> III in that
order, so the label is very nearly a deterministic function of position in the
session. Any feature that increases monotonically with time is therefore a clock,
and a clock solves this task without looking at the person at all.

Measured, cross-user, ONE feature each, logistic regression:

    widx       (position / session length)   AUC 0.998
    elapsed    (window index)                AUC 0.998
    cum_move   (cumulative distance moved)   AUC 0.996
    dist_base  (distance from baseline)      AUC 0.988
    step       (distance from previous)      AUC 0.563   <- not monotone

That 0.998 is not error detection, it is the experiment schedule. `widx` also
divides by the total session length, which is look-ahead a deployed robot could
not have. All four monotone scalars have been REMOVED from the feature builder
and the sweep; only `step` survives. If you ever want to demonstrate the effect
in the paper, reintroduce them deliberately and label them as an upper bound on
schedule leakage, not as a result.

The vector deltas (delta_base, delta_prev, drift) are differences, not clocks, so
they are kept -- but the same suspicion should be applied to anything new.

CALIBRATION AND THE NOISE FLOOR -- read before quoting any number here
----------------------------------------------------------------------
Two things inflate these AUCs relative to the flat v4 sweep. Neither is a bug,
but both must be stated.

1. THE BASELINE IS NOT 0.5. Replacing the embeddings with pure Gaussian noise,
   keeping the session structure identical, still gives:

        window_sec  current-only   all-deltas
             1         0.554         0.554
             3         0.581         0.599
             5         0.589         0.610
            10         0.474         0.571

   The first windows of a session have structurally smaller deltas (delta_prev
   and drift are exactly 0 at window 0; delta_base is near 0 while the window is
   still inside the baseline), and the first windows are always the survey, i.e.
   always label 0. So position leaks through the feature magnitudes even with no
   signal at all. Real minus noise is the honest effect size:

        window_sec=5, all deltas:  0.923 real - 0.610 noise = 0.313
        window_sec=5, current:     0.841 real - 0.589 noise = 0.252

2. delta_base IS PER-USER CALIBRATION. It subtracts the test participant's own
   first `baseline_sec` seconds, so the model has seen that person before it
   predicts. It is deployable (a robot can watch for 5 s first), but it is NOT
   the "no per-user calibration" setting the paper claims. At window_sec=1 it is
   worth +0.12 AUC (0.795 -> 0.914) -- most of the apparent gain. Report
   use_current-only runs as the zero-shot number and delta runs as calibrated.
"""

import argparse
import os
import sys

import numpy as np
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "common"))  # eval_utils needs get_all_metrics

import hier_data as H  # noqa: E402
from eval_utils import _core_metrics, participant_bootstrap  # noqa: E402


def escalation_features(session, baseline_win, cfg):
    """Build the per-window feature matrix for one session."""
    W = session["X"]                      # (n_windows, window_sec, n_feat)
    cur = W.mean(axis=1)                  # (n_windows, n_feat)
    n = len(cur)
    base = cur[:max(1, baseline_win)].mean(axis=0, keepdims=True)

    prev = np.vstack([cur[:1], cur[:-1]])
    drift = np.vstack([cur[:1]] + [cur[:i].mean(axis=0, keepdims=True)
                                   for i in range(1, n)])
    step = np.linalg.norm(cur - prev, axis=1)

    blocks = []
    if cfg.use_current:
        blocks.append(cur)
    if cfg.use_delta_base:
        blocks.append(cur - base)
    if cfg.use_delta_prev:
        blocks.append(cur - prev)
    if cfg.use_drift:
        blocks.append(cur - drift)
    if cfg.use_scalars:
        # `step` only. See the LEAKAGE note in the module docstring: every
        # monotone-in-time scalar (widx, elapsed, cum_move, dist_base) predicts
        # the label almost perfectly on its own and must not be swept.
        blocks.append(step[:, None])
    if not blocks:                        # never hand back an empty matrix
        blocks.append(cur)
    return np.hstack(blocks).astype("float32")


def make_clf(cfg):
    if cfg.model_type == "logreg":
        clf = LogisticRegression(max_iter=3000, C=cfg.C)
    elif cfg.model_type == "sgd":
        clf = SGDClassifier(loss="log_loss", alpha=cfg.alpha, max_iter=3000)
    elif cfg.model_type == "rf":
        return RandomForestClassifier(n_estimators=cfg.n_estimators,
                                      max_depth=cfg.max_depth,
                                      class_weight="balanced", random_state=42)
    elif cfg.model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(cfg.units_l1,), max_iter=500,
                            random_state=42)
    else:
        raise ValueError(cfg.model_type)
    return make_pipeline(StandardScaler(), clf)


def train():
    wandb.init()
    cfg = wandb.config

    info, X = H.load_frames(cfg.modality, cfg.gemini_source, cfg.gemini_dims,
                            feature_set=getattr(cfg, "feature_set", "full"),
                            binary_multiclass=cfg.binary_multiclass)
    Xs, yb, ym, parts = H.to_seconds(info, X)
    y = yb if cfg.binary_multiclass == "binary" else ym
    wandb.log({"n_seconds": len(Xs), "n_features_raw": Xs.shape[1]})

    sessions = H.build_sessions(Xs, y, parts, cfg.window_sec, cfg.stride_sec)
    # Noise-input control -- see hier_model.py. The docstring quotes a 0.55-0.61
    # floor measured ad hoc; this makes it a swept, logged condition.
    if getattr(cfg, "noise_input", False):
        _rng = np.random.default_rng(1000 + int(getattr(cfg, "seed", 0)))
        for _s in sessions:
            _s["X"] = _rng.standard_normal(_s["X"].shape).astype("float32")
    baseline_win = max(1, cfg.baseline_sec // max(1, cfg.stride_sec))

    feats = {s["part"]: (escalation_features(s, baseline_win, cfg), s["y"])
             for s in sessions}
    wandb.log({"n_features_escalation": next(iter(feats.values()))[0].shape[1]})

    n_classes = int(y.max()) + 1
    folds = H.participant_folds(parts)
    fold_metrics, P, T, G, PR = [], [], [], [], []

    for fold, (tr_p, _va_p, te_p) in enumerate(folds):
        tr = [p for p in tr_p if p in feats]
        te = [p for p in te_p if p in feats]
        if not tr or not te:
            continue
        Xtr = np.vstack([feats[p][0] for p in tr])
        ytr = np.concatenate([feats[p][1] for p in tr])
        Xte = np.vstack([feats[p][0] for p in te])
        yte = np.concatenate([feats[p][1] for p in te])
        gte = np.concatenate([np.full(len(feats[p][1]), p) for p in te])
        if len(np.unique(ytr)) < 2:
            continue

        clf = make_clf(cfg).fit(Xtr, ytr)
        raw = clf.predict_proba(Xte)
        # A fold whose training split is missing a class returns fewer columns.
        # Scatter into fixed-width columns by class id so folds stack, and so the
        # pooled AUC below is computed over a consistent label space.
        classes = (clf.classes_ if hasattr(clf, "classes_")
                   else clf[-1].classes_).astype(int)
        proba = np.zeros((len(Xte), n_classes), dtype=float)
        proba[:, classes] = raw
        pred = classes[raw.argmax(axis=1)]

        m = _core_metrics(pred, yte, proba)
        fold_metrics.append(m)
        # Same key shape as v4/v5 (t<fold>_test_<metric> / avg_test_<metric>) so
        # the three sweeps can be compared in one W&B view.
        wandb.log({f"t{fold}_test_{k}": v for k, v in m.items()})
        print(f"fold {fold}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        P.append(pred); T.append(yte); G.append(gte); PR.append(proba)

    if not fold_metrics:
        wandb.log({"status": "no_valid_folds"})
        return

    wandb.log({f"avg_test_{k}": float(np.mean([m[k] for m in fold_metrics]))
               for k in fold_metrics[0]})
    P, T, G = np.concatenate(P), np.concatenate(T), np.concatenate(G)
    PR = np.vstack(PR)
    # The evaluation unit here is already the window, so the pooled estimate and
    # its participant bootstrap take the sec_test_* names v4/v5 use.
    # PR (not None) is essential: _core_metrics returns AUC = NaN without
    # probabilities, which is what made avg_sec_test_auc NaN on the first runs.
    wandb.log({f"avg_sec_test_{k}": v for k, v in _core_metrics(P, T, PR).items()})
    wandb.log({f"avg_sec_test_{k}": v for k, v in
               participant_bootstrap(P, T, G, PR, n_bootstrap=500).items()})
    wandb.log({"avg_n_eval_units": int(len(T)),
               "avg_n_eval_participants": int(len(set(G)))})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="brirl_hier_escalation_v6")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--count", type=int, default=None)
    args = ap.parse_args()

    sweep = {
        "method": "random",
        "name": "brirl_hier_escalation_v6_linear",
        "parameters": {
            "binary_multiclass": {"values": ["binary", "multiclass"]},
            # Control: identical structure, features replaced by noise.
            "noise_input": {"values": [False, True]},
            "seed": {"values": [0, 1, 2]},
            "model_type": {"values": ["logreg", "sgd", "rf", "mlp"]},
            # --- which escalation blocks are switched on --------------------
            # use_current alone == the flat control.
            "use_current":    {"values": [True, False]},
            "use_delta_base": {"values": [True, False]},
            "use_delta_prev": {"values": [True, False]},
            "use_drift":      {"values": [True, False]},
            "use_scalars":    {"values": [True, False]},
            # Calibration budget. The survey block is only ~10 s, so a 10 s
            # baseline would consume essentially all of the neutral period and
            # leave nothing to classify. OLD: [3, 5, 10].
            "baseline_sec":   {"values": [2, 3, 5]},
            # --- temporal granularity (seconds; data resampled to 1 Hz) ------
            # Sessions are only ~29 s after the recovery block is dropped (min 23,
            # max 43), so a long window is most of the interaction. Fraction of
            # windows that contain a SINGLE label:
            #     window_sec   binary   multiclass(4-class)
            #          1        100%        100%
            #          2         97%         90%
            #          3         93%         79%
            #          5         85%         54%
            #         10         58%         10%   <- removed
            # Error I averages only ~5.5 s per participant, so a 10 s window
            # cannot sit inside it at all: it gets its last second's label while
            # mostly containing something else. That is mislabelling, not
            # smoothing. OLD: [3, 5, 10].
            "window_sec": {"values": [1, 2, 3, 5]},
            # Stride is clamped to the window at build time; 5 was only useful
            # for window_sec=10. OLD: [1, 2, 5].
            "stride_sec": {"values": [1, 2]},
            # --- model hyperparameters ---------------------------------------
            "C": {"values": [0.1, 1.0, 10.0]},
            "alpha": {"values": [1e-5, 1e-4, 1e-3]},
            "n_estimators": {"values": [200, 500]},
            "max_depth": {"values": [5, 10, 20]},
            "units_l1": {"values": [64, 128]},
            # --- features -----------------------------------------------------
            # Conventional-feature source. Curated sets are one undivided block;
            # they still compose with text/cosine/gemini (see feature_sets.py).
            "feature_set": {"values": ["full", "curated_v5", "rf", "selectkbest"]},
            "gemini_source": {"values": ["visual", "visual_audio"]},
            "gemini_dims": {"values": [128, 256, 768]},
            "modality": {"values": [
                "gemini", "facial", "audio", "pose",
                "facial_audio", "pose_facial_audio",
                "facial_gemini", "audio_gemini", "pose_gemini",
                "facial_audio_gemini", "pose_facial_audio_gemini",
            ]},
        },
    }
    print(sweep)
    sid = wandb.sweep(sweep=sweep, project=args.project, entity=args.entity)
    wandb.agent(sid, function=train, count=args.count,
                entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
