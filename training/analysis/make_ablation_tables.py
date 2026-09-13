#!/usr/bin/env python3
"""Per-track modality-ablation tables from the uncapped v4-v6 W&B exports.

Replaces the single markdown file (corrections/HRI27_modality_ablation.md) with
one CSV per track x task, each row an ablation cell carrying the winning run's
FULL configuration, so a row can be written straight into the paper and traced
back to the exact W&B run.

Conventions are inherited from corrections/make_results_tables.py so these
numbers stay comparable to HRI27_results_by_track.csv:

  * the reported metric is SECOND-LEVEL (`avg_sec_test_*`), not frame-level;
  * deep AUC uses `avg_sec_test_auc` unless it is the 0.5 placeholder left by
    Bug B, in which case it falls back to frame-level and is marked 'frame';
  * one row per input cell = the best FINISHED run in that cell by second-level
    macro-F1, reported with the spread over the runs in the cell.

What is new here:

  * `--deep_cutoff`: deep runs created before the Bug A/B fix are dropped. Bug A
    (create_sequences labels a window by its LAST frame) cost long-context runs
    the entire pre-failure block -- 10 of 93 deep rows lost it completely, and
    all three top detection-stage runs were scored on a test set with no class 0
    in it. Bug B (aggregate_blocks renormalising a single sigmoid column by
    itself) pinned second-level AUC to exactly 0.5. Neither is recoverable from
    the logged summary, so those runs are removed rather than annotated. Linear
    runs are untouched: neither bug can reach them (sklearn predict_proba
    returns two columns for binary, and no sequence construction happens).
  * every hyperparameter the sweep set is carried through, per model family;
  * run id + URL, so any row can be opened in W&B.

Usage
-----
    python3 make_ablation_tables.py --raw analysis_results/raw \
        --out analysis_results --deep_cutoff 2026-09-11T04:28:11
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MP = {"facial": "F", "audio": "A", "pose": "P", "text": "T", "cosine": "C", "gemini": "G"}

# The 14 rows of HRI27_modality_ablation.md, in order, as (label, predicate).
# Kept identical so the new CSVs line up with the table already in the draft.
ABLATION_ROWS = [
    ("F",                              dict(modality="facial")),
    ("A",                              dict(modality="audio")),
    ("P",                              dict(modality="pose")),
    ("T",                              dict(modality="text")),
    ("C",                              dict(modality="cosine")),
    ("F+A",                            dict(modality="facial_audio")),
    ("F+A+P",                          dict(modality="pose_facial_audio")),
    ("F+A+P+T",                        dict(modality="pose_facial_audio_text")),
    ("F+A+P+C",                        dict(modality="pose_facial_audio_cosine")),
    ("best engineered (any set)",      dict(kind="engineered")),
    ("G [video]",                      dict(modality="gemini", gemini_source="visual")),
    ("G [video+audio]",                dict(modality="gemini", gemini_source="visual_audio")),
    ("best engineered+G [video]",      dict(kind="mixed", gemini_source="visual")),
    ("best engineered+G [video+audio]", dict(kind="mixed", gemini_source="visual_audio")),
]

# Hyperparameters carried into the output, grouped so the column order reads
# sensibly. Missing ones stay blank -- an rf row has no `kernel`.
SHARED = ["feature_set", "dataset", "window", "stride", "aggregation", "agg_features",
          "test_window_size", "test_stride", "feature_randomizer",
          "gemini_source", "gemini_dims"]
PER_MODEL = ["n_estimators", "max_depth",            # rf
             "alpha", "max_iter", "tol",             # sgd
             "C", "kernel", "gamma",                 # svm
             "n_neighbors",                          # knn
             "mlp_hidden_layer_sizes", "mlp_activation",   # mlp
             "sequence_length", "fusion_type", "units", "num_layers", "dropout",
             "learning_rate", "batch_size", "epochs", "patience",
             "num_heads", "ff_dim", "l2_reg", "label_smoothing", "optimizer"]


def has_g(m):
    return "gemini" in str(m)


def is_engineered(m):
    return not has_g(m)


def is_mixed(m):
    """Engineered blocks AND embeddings (i.e. gemini, but not gemini alone)."""
    return has_g(m) and str(m) != "gemini"


def folds(r, metric):
    """Per-fold values, tolerating the three key spellings the scripts have used."""
    out = []
    for k in range(5):
        for c in (f"t{k}_{metric}", f"fold_{k}_metrics.{metric}", f"fold_{k}_{metric}"):
            if c in r.index and pd.notna(r[c]):
                out.append(float(r[c]))
                break
    return np.array(out)


def prep(path, deep, track, cutoff=None):
    """Load one project export and normalise the columns that differ by family."""
    if not os.path.exists(path):
        print(f"  (missing: {path})")
        return pd.DataFrame()
    d = pd.read_csv(path, low_memory=False)
    n0 = len(d)
    d = d[d.State == "finished"].copy()
    n_fin = len(d)

    dropped_time = 0
    if deep and cutoff is not None and "Created" in d:
        ts = pd.to_datetime(d["Created"], errors="coerce", utc=True)
        keep = ts >= pd.Timestamp(cutoff, tz="UTC")
        dropped_time = int((~keep).sum())
        d = d[keep].copy()

    if not len(d):
        print(f"  {os.path.basename(path)}: {n0} rows, {n_fin} finished, "
              f"{dropped_time} dropped by time, 0 usable")
        return d

    if deep:
        d["task"] = d.get("class")
        d["mtype"] = d.get("model")
        d["family"] = "deep"
        sec = pd.to_numeric(d.get("avg_sec_test_auc"), errors="coerce")
        frm = pd.to_numeric(d.get("avg_test_auc"), errors="coerce")
        sec_ok = sec.notna() & (sec != 0.5)
        d["auc"] = np.where(sec_ok, sec, frm)
        d["auc_unit"] = np.where(sec_ok, "second", "frame")
        dims = pd.to_numeric(d.get("gemini_dims"), errors="coerce")
        d["dims"] = dims.astype("Int64").astype(str).str.replace("3072", "full", regex=False)
    else:
        d["task"] = d.get("binary_multiclass")
        d["mtype"] = d.get("model_type")
        d["family"] = "window-level"
        d["auc"] = pd.to_numeric(d.get("avg_sec_test_auc"), errors="coerce")
        d["auc_unit"] = "second"
        d["dims"] = d.get("gemini_embedding").astype(str).str.replace("d", "", regex=False)

    d["f1"] = pd.to_numeric(d.get("avg_sec_test_f1"), errors="coerce")
    d["track"] = track
    d = d.dropna(subset=["f1"])
    print(f"  {os.path.basename(path)}: {n0} rows, {n_fin} finished, "
          f"{dropped_time} dropped by time, {len(d)} with a second-level F1")
    return d


def cell(sub, label):
    """One ablation row: the best run in the cell plus the cell's spread."""
    if not len(sub):
        return dict(inputs=label, n_runs=0, model_family="", model="", modality="",
                    F1="", F1_fold_std="", F1_ci_low="", F1_ci_high="", AUC="",
                    AUC_unit="", FNR="", Accuracy="", Precision="", Recall="",
                    median_F1_in_cell="", top5_mean_F1="", F1_per_fold="",
                    run_name="", run_id="", sweep="", run_url="", created="",
                    note="no runs in this cell")
    r = sub.loc[sub.f1.idxmax()]
    f1f = folds(r, "sec_test_f1")
    aucf = folds(r, "sec_test_auc") if r.auc_unit == "second" else folds(r, "test_auc")
    g = has_g(r.modality)

    def get(k, default=""):
        v = r.get(k, default)
        if isinstance(v, float) and np.isnan(v):
            return ""
        return v

    def rnd(k):
        v = pd.to_numeric(r.get(k), errors="coerce")
        return round(float(v), 3) if pd.notna(v) else ""

    row = dict(
        inputs=label,
        n_runs=len(sub),
        model_family=r.family,
        model=r.mtype,
        modality=r.modality,
        F1=round(float(r.f1), 3),
        F1_fold_std=round(float(f1f.std(ddof=1)), 3) if len(f1f) > 1 else "",
        F1_ci_low=rnd("avg_sec_test_f1_ci_low"),
        F1_ci_high=rnd("avg_sec_test_f1_ci_high"),
        AUC=round(float(r.auc), 3) if pd.notna(r.auc) else "",
        AUC_unit=r.auc_unit,
        AUC_fold_std=round(float(aucf.std(ddof=1)), 3) if len(aucf) > 1 else "",
        FNR=rnd("avg_sec_test_fnr"),
        Accuracy=rnd("avg_sec_test_accuracy"),
        Precision=rnd("avg_sec_test_precision"),
        Recall=rnd("avg_sec_test_recall"),
        F1_per_fold=";".join(f"{x:.3f}" for x in f1f),
        median_F1_in_cell=round(float(sub.f1.median()), 3),
        top5_mean_F1=round(float(sub.f1.nlargest(5).mean()), 3),
    )

    # How safe is "the winner"? In v4 detection-binary P the best deep run beats
    # the best window-level run by 0.00002 -- naming a family winner there would
    # be reading noise, so the margin travels with the row.
    top2 = sub.f1.nlargest(2).tolist()
    row["runner_up_F1"] = round(float(top2[1]), 3) if len(top2) > 1 else ""
    row["margin_over_runner_up"] = round(float(top2[0] - top2[1]), 5) if len(top2) > 1 else ""
    other = sub[sub.family != r.family]
    if len(other):
        bo = float(other.f1.max())
        row["best_other_family"] = other.loc[other.f1.idxmax(), "family"]
        row["best_other_family_F1"] = round(bo, 3)
        row["margin_over_other_family"] = round(float(r.f1) - bo, 5)
    else:
        row["best_other_family"] = ""
        row["best_other_family_F1"] = ""
        row["margin_over_other_family"] = ""
    for k in SHARED:
        if k in ("gemini_source", "gemini_dims"):
            row[k] = (get("gemini_source") if k == "gemini_source" else r.get("dims", "")) if g else ""
        else:
            row[k] = get(k)
    for k in PER_MODEL:
        row[k] = get(k)
    note = ""
    m = row["margin_over_other_family"]
    if m != "" and abs(m) < 0.005:
        note = (f"family winner is a tie: best {row['best_other_family']} run is "
                f"{row['best_other_family_F1']} (margin {m:+.5f}); do not read "
                f"this cell as one family beating the other")
    row.update(run_name=get("Name"), run_id=get("ID"), sweep=get("Sweep"),
               run_url=get("URL"), created=get("Created"), note=note)
    return row


def build(D, track, task):
    s = D[(D.track == track) & (D.task == task)]
    rows = []
    for label, spec in ABLATION_ROWS:
        kind = spec.get("kind")
        if kind == "engineered":
            sub = s[s.modality.map(is_engineered)]
        elif kind == "mixed":
            sub = s[s.modality.map(is_mixed) & (s.gemini_source == spec["gemini_source"])]
        else:
            sub = s[s.modality == spec["modality"]]
            if "gemini_source" in spec:
                sub = sub[sub.gemini_source == spec["gemini_source"]]
        r = cell(sub, label)
        r["track"], r["task"] = track, task
        rows.append(r)
    out = pd.DataFrame(rows)
    front = ["track", "task", "inputs", "n_runs", "model_family", "model", "modality"]
    return out[front + [c for c in out.columns if c not in front]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="analysis_results/raw")
    ap.add_argument("--out", default="analysis_results")
    ap.add_argument("--deep_cutoff", default="2026-09-11T04:28:11",
                    help="UTC; deep runs created before this are dropped (Bug A/B)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = lambda n: os.path.join(a.raw, n + ".csv")

    print("loading exports:")
    D = pd.concat([
        prep(R("brirl_gemini_v4_linear"), False, "detection"),
        prep(R("brirl_gemini_v4_deep"), True, "detection", a.deep_cutoff),
        prep(R("brirl_gemini_v5_recovery_linear"), False, "recovery"),
        prep(R("brirl_gemini_v5_recovery_deep"), True, "recovery", a.deep_cutoff),
    ], ignore_index=True)
    print(f"pooled usable runs: {len(D)}")

    D.to_csv(os.path.join(a.out, "all_runs_clean.csv"), index=False)

    made = []
    for track, task, name in [("detection", "binary", "ablation_v4_detection_binary"),
                              ("detection", "multiclass", "ablation_v4_detection_multiclass"),
                              ("recovery", "binary", "ablation_v5_recovery_binary"),
                              ("recovery", "multiclass", "ablation_v5_recovery_multiclass")]:
        t = build(D, track, task)
        p = os.path.join(a.out, name + ".csv")
        t.to_csv(p, index=False)
        filled = int((t.n_runs > 0).sum())
        print(f"{name}: {filled}/14 cells filled, "
              f"{int(t.n_runs.sum())} runs -> {p}")
        made.append(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
