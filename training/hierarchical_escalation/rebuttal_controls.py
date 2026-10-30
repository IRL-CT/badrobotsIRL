#!/usr/bin/env python3
"""HRI'27 rebuttal: temporal controls, matched block ladder and baseline-length
sweep for the relative-feature arm (escalation_linear.py), on fixed configs.

Why a separate driver
---------------------
escalation_linear.py is a W&B random-search agent. The rebuttal asks for a
handful of *deterministic* comparisons on the two best published configs, each
with a different manipulation of the session timeline, plus pooled confusion
matrices that the sweep never logged. Everything here is computed by the same
functions the sweep used (hier_data loading / 1 Hz collapse / participant folds,
eval_utils metrics); only the session timeline and the feature blocks are
manipulated, and every manipulation is a named, seeded condition in one CSV.

Rebuttal items covered (HRI27_rebuttal_triage.xlsx)
---------------------------------------------------
T2  temporal confound. Controls, all applied to train, validation AND test
    sessions with a fresh draw per fold and seed:
      none           the published pipeline (reference, same driver)
      noise          Gaussian noise of the same shape (the paper's control)
      onset5/onset10 random onset: drop a random 0-R s prefix of every session,
                     so sessions no longer share a time origin; delta_base is
                     then taken from the first *observed* seconds
      label_shift    R2's control: cyclic shift of the label sequence against
                     the features within each session by a random offset of at
                     least 5 s; phase lengths and order are preserved, the
                     alignment is destroyed
      phase_shuffle  R1's control: permute the order of the seconds inside each
                     phase (contiguous same-label run) before windowing and
                     before delta_prev / drift are computed; phase membership
                     and phase order are preserved, within-phase dynamics are
                     destroyed
T1  matched ladder: current only / +delta_prev,drift / +delta_base on the SAME
    modality, model and window, so the comparison holds everything but the
    feature framing constant. T2b repeats the ladder under the two random-onset
    controls, so the absolute and relative rows are compared on identical
    shifted sessions (separates "the test set got harder" from "the relative
    features lost their anchor").
T11 baseline length: the sweep's `baseline_sec` is converted to WINDOWS
    (baseline_sec // stride_sec), so with w=5 s, stride 2 s the values 2 and 3
    both mean "the first 5 s window". `--baseline_mode seconds` computes the
    baseline from the first b raw seconds instead, for b in {1,2,3,5}.
T10 pooled 4x4 (stage) and 2x2 (detection) confusion matrices, second/window
    level, pooled over the five test folds, for every run.

Usage
-----
    python3 rebuttal_controls.py --plan            # list the runs
    python3 rebuttal_controls.py                   # run everything, resumable
    python3 rebuttal_controls.py --only T2 --seeds 0
Output: training/analysis/rebuttal/relative_controls.csv (one row per run,
confusion matrices as JSON), appended after every run.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "common"))

import hier_data as H  # noqa: E402
from eval_utils import _core_metrics, participant_bootstrap  # noqa: E402

OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "analysis", "rebuttal"))
OUT = os.path.join(OUT_DIR, "relative_controls.csv")

# ----------------------------------------------------------------------------
# The two published best relative-feature runs (main-text Table 5), copied from
# their W&B configs (project brirl_hier_escalation_v6).
#   stage_best   serene-sweep-42  (xbxdi8fv)  F1 0.631  n_features_raw 287
#                = 31 curated + 256-D video+audio embedding
#   detect_best  fast-sweep-2151  (jmtc3kqn)  F1 0.959  n_features_raw 25
#                = eGeMAPS acoustic block ONLY (modality "audio"; the
#                gemini_* keys are inert sweep parameters for this modality)
# ----------------------------------------------------------------------------
CONFIGS = {
    "stage_best": dict(
        wandb_run="serene-sweep-42/xbxdi8fv", binary_multiclass="multiclass",
        model_type="rf", n_estimators=200, max_depth=20,
        modality="pose_gemini", feature_set="curated_v5",
        gemini_source="visual_audio", gemini_dims=256,
        window_sec=5, stride_sec=2, baseline_sec=2,
        use_current=False, use_delta_base=True, use_delta_prev=True,
        use_drift=True, use_scalars=False,
        C=1.0, alpha=1e-4, units_l1=64),
    "detect_best": dict(
        wandb_run="fast-sweep-2151/jmtc3kqn", binary_multiclass="binary",
        model_type="logreg", C=10.0,
        modality="audio", feature_set="full",
        gemini_source="visual_audio", gemini_dims=768,
        window_sec=5, stride_sec=2, baseline_sec=5,
        use_current=False, use_delta_base=True, use_delta_prev=True,
        use_drift=True, use_scalars=False,
        n_estimators=200, max_depth=20, alpha=1e-4, units_l1=64),
}

BLOCK_LADDER = {
    "current":           dict(use_current=True,  use_delta_base=False, use_delta_prev=False, use_drift=False),
    "dprev_drift":       dict(use_current=False, use_delta_base=False, use_delta_prev=True,  use_drift=True),
    "dprev_drift_dbase": dict(use_current=False, use_delta_base=True,  use_delta_prev=True,  use_drift=True),
}

CONTROLS = ["none", "noise", "onset5", "onset10", "label_shift", "phase_shuffle"]
MIN_SHIFT_S = 5


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


# ----------------------------------------------------------------------------
# Session construction with the timeline manipulations
# ----------------------------------------------------------------------------
def manipulate_seconds(Xp, yp, control, rng):
    """Apply one control to one participant's 1 Hz stream. Returns (X, y, info)."""
    info = {}
    n = len(yp)
    if control == "none":
        return Xp, yp, info
    if control == "noise":
        return rng.standard_normal(Xp.shape).astype("float32"), yp, info
    if control.startswith("onset"):
        R = int(control[len("onset"):])
        cut = int(rng.integers(0, R + 1))
        info["cut_s"] = cut
        return Xp[cut:], yp[cut:], info
    if control == "label_shift":
        hi = n - MIN_SHIFT_S
        k = int(rng.integers(MIN_SHIFT_S, hi + 1)) if hi >= MIN_SHIFT_S else n // 2
        info["shift_s"] = k
        return Xp, np.roll(yp, k), info
    if control == "phase_shuffle":
        X = Xp.copy()
        # contiguous same-label runs = phases (survey, Error I, II, III)
        bounds = np.flatnonzero(np.diff(yp) != 0) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [n]])
        for s, e in zip(starts, ends):
            if e - s > 1:
                X[s:e] = X[s:e][rng.permutation(e - s)]
        info["n_phases"] = int(len(starts))
        return X, yp, info
    raise ValueError(control)


def build_sessions(Xs, y, parts, window_sec, stride_sec, control, rng):
    """hier_data.build_sessions plus the raw 1 Hz stream and a control."""
    sessions, infos = [], {}
    for p in pd.unique(parts):
        m = parts == p
        Xp, yp, info = manipulate_seconds(Xs[m], y[m], control, rng)
        infos[str(p)] = info
        n = len(Xp)
        starts = list(range(0, n - window_sec + 1, stride_sec))
        if not starts:
            continue
        wX = np.stack([Xp[s:s + window_sec] for s in starts]).astype("float32")
        wy = np.array([yp[s + window_sec - 1] for s in starts])
        sessions.append({"X": wX, "y": wy, "raw": Xp, "part": p})
    return sessions, infos


def escalation_features(session, cfg, baseline_mode, baseline_sec):
    """escalation_linear.escalation_features with a seconds-based baseline option."""
    W = session["X"]
    cur = W.mean(axis=1)
    n = len(cur)
    if baseline_mode == "windows":           # the sweep's convention
        bw = max(1, baseline_sec // max(1, cfg.stride_sec))
        base = cur[:max(1, bw)].mean(axis=0, keepdims=True)
    elif baseline_mode == "seconds":         # first b observed seconds
        base = session["raw"][:max(1, baseline_sec)].mean(axis=0, keepdims=True)
    else:
        raise ValueError(baseline_mode)

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
        blocks.append(step[:, None])
    if not blocks:
        blocks.append(cur)
    return np.hstack(blocks).astype("float32")


def make_clf(cfg, seed, n_jobs):
    if cfg.model_type == "logreg":
        clf = LogisticRegression(max_iter=3000, C=cfg.C)
    elif cfg.model_type == "sgd":
        clf = SGDClassifier(loss="log_loss", alpha=cfg.alpha, max_iter=3000, random_state=seed)
    elif cfg.model_type == "rf":
        # The sweep fixed random_state=42; the seed here varies the forest so the
        # three seeds are genuine replicates. n_jobs does not change the result.
        return RandomForestClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                                      class_weight="balanced", random_state=42 + seed,
                                      n_jobs=n_jobs)
    elif cfg.model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(cfg.units_l1,), max_iter=500, random_state=42 + seed)
    else:
        raise ValueError(cfg.model_type)
    return make_pipeline(StandardScaler(), clf)


# ----------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------
_DATA_CACHE = {}


def load_seconds(cfg):
    key = (cfg.modality, cfg.gemini_source, cfg.gemini_dims, cfg.feature_set, cfg.binary_multiclass)
    if key not in _DATA_CACHE:
        info, X = H.load_frames(cfg.modality, cfg.gemini_source, cfg.gemini_dims,
                                feature_set=cfg.feature_set, binary_multiclass=cfg.binary_multiclass)
        _DATA_CACHE[key] = H.to_seconds(info, X)
    return _DATA_CACHE[key]


def run_one(cfg, control, baseline_mode, baseline_sec, seed, n_jobs, n_boot):
    Xs, yb, ym, parts = load_seconds(cfg)
    y = yb if cfg.binary_multiclass == "binary" else ym
    n_classes = int(y.max()) + 1
    folds = H.participant_folds(parts)

    fold_metrics, P, T, G, PR, infos_all = [], [], [], [], [], {}
    for fold, (tr_p, _va_p, te_p) in enumerate(folds):
        # fresh draw per fold and seed, applied to every session (train, val, test)
        rng = np.random.default_rng(10_000 * (seed + 1) + fold)
        sessions, infos = build_sessions(Xs, y, parts, cfg.window_sec, cfg.stride_sec, control, rng)
        infos_all[fold] = infos
        feats = {s["part"]: (escalation_features(s, cfg, baseline_mode, baseline_sec), s["y"])
                 for s in sessions}
        tr = [p for p in tr_p if p in feats]
        te = [p for p in te_p if p in feats]
        if not tr or not te:
            continue
        Xtr = np.vstack([feats[p][0] for p in tr]); ytr = np.concatenate([feats[p][1] for p in tr])
        Xte = np.vstack([feats[p][0] for p in te]); yte = np.concatenate([feats[p][1] for p in te])
        gte = np.concatenate([np.full(len(feats[p][1]), p) for p in te])
        if len(np.unique(ytr)) < 2:
            continue
        clf = make_clf(cfg, seed, n_jobs).fit(Xtr, ytr)
        raw = clf.predict_proba(Xte)
        classes = (clf.classes_ if hasattr(clf, "classes_") else clf[-1].classes_).astype(int)
        proba = np.zeros((len(Xte), n_classes)); proba[:, classes] = raw
        pred = classes[raw.argmax(axis=1)]
        fold_metrics.append(_core_metrics(pred, yte, proba))
        P.append(pred); T.append(yte); G.append(gte); PR.append(proba)

    P, T, G, PR = np.concatenate(P), np.concatenate(T), np.concatenate(G), np.vstack(PR)
    pooled = _core_metrics(P, T, PR)
    boot = participant_bootstrap(P, T, G, PR, n_bootstrap=n_boot)
    labels = list(range(n_classes))
    cm = confusion_matrix(T, P, labels=labels)
    row = {
        "avg_test_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "avg_test_auc": float(np.nanmean([m["auc"] for m in fold_metrics])),
        "fold_f1": json.dumps([round(m["f1"], 4) for m in fold_metrics]),
        "sec_f1": pooled["f1"], "sec_auc": pooled["auc"], "sec_fnr": pooled["fnr"],
        "sec_f1_ci_low": boot.get("f1_ci_low"), "sec_f1_ci_high": boot.get("f1_ci_high"),
        "sec_f1_boot_std": boot.get("f1_boot_std"),
        "n_eval_units": int(len(T)), "n_eval_participants": int(len(set(G))),
        "n_features": int(Xtr.shape[1]),
        "confusion": json.dumps(cm.tolist()),
        "control_info": json.dumps({str(k): v for k, v in infos_all.items()})
        if control not in ("none", "noise") else "",
    }
    return row


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
def make_plan(seeds):
    plan = []
    for cname, base in CONFIGS.items():
        for ctl in CONTROLS:                                        # T2 (+T10 from "none")
            for s in seeds:
                plan.append(dict(item="T2", config=cname, control=ctl, blocks="as_published",
                                 baseline_mode="windows", baseline_sec=base["baseline_sec"], seed=s))
        for bname in BLOCK_LADDER:                                  # T1
            for s in seeds:
                plan.append(dict(item="T1", config=cname, control="none", blocks=bname,
                                 baseline_mode="windows", baseline_sec=base["baseline_sec"], seed=s))
        for ctl in ("onset5", "onset10"):                           # T2b: ladder under random onset
            for bname in BLOCK_LADDER:                              # (absolute vs relative on the SAME shifted sessions)
                for s in seeds:
                    plan.append(dict(item="T2b", config=cname, control=ctl, blocks=bname,
                                     baseline_mode="windows", baseline_sec=base["baseline_sec"], seed=s))
        for b in (1, 2, 3, 5):                                      # T11
            for s in seeds:
                plan.append(dict(item="T11", config=cname, control="none", blocks="as_published",
                                 baseline_mode="seconds", baseline_sec=b, seed=s))
    return plan


def plan_key(p):
    return (p["config"], p["control"], p["blocks"], p["baseline_mode"], int(p["baseline_sec"]), int(p["seed"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None, help="subset of items: T1 T2 T11")
    ap.add_argument("--configs", nargs="*", default=None, help="subset of CONFIGS keys")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    plan = make_plan(a.seeds)
    if a.only:
        plan = [p for p in plan if p["item"] in set(a.only)]
    if a.configs:
        plan = [p for p in plan if p["config"] in set(a.configs)]
    if a.plan:
        for p in plan:
            print(p)
        print(len(plan), "runs")
        return 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = {plan_key(r) for r in prev.to_dict("records")}
        print(f"resuming: {len(done)} runs already in {a.out}", flush=True)

    for i, p in enumerate(plan):
        if plan_key(p) in done:
            continue
        cfg = Cfg(CONFIGS[p["config"]])
        if p["blocks"] != "as_published":
            cfg.update(BLOCK_LADDER[p["blocks"]])
        t0 = time.time()
        row = run_one(cfg, p["control"], p["baseline_mode"], p["baseline_sec"], p["seed"], a.n_jobs, a.n_boot)
        row = {**p, "wandb_run": cfg.wandb_run, "model_type": cfg.model_type, "modality": cfg.modality,
               "feature_set": cfg.feature_set, "gemini_dims": cfg.gemini_dims if "gemini" in cfg.modality else "",
               "window_sec": cfg.window_sec, "stride_sec": cfg.stride_sec, **row,
               "runtime_s": round(time.time() - t0, 1)}
        pd.DataFrame([row]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        done.add(plan_key(p))
        print(f"[{i + 1}/{len(plan)}] {p['item']} {p['config']} {p['control']} {p['blocks']} "
              f"{p['baseline_mode']}={p['baseline_sec']} seed={p['seed']}: "
              f"F1 {row['sec_f1']:.3f} AUC {row['sec_auc']:.3f} ({row['runtime_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
