#!/usr/bin/env python3
"""Time-matched evaluation (§5.4, Fig. 4 right panel).

In a fixed protocol the label is nearly a function of elapsed time. This script recomputes second-level
macro-F1 and AUC (a) on all seconds and (b) only inside the elapsed-time band where both classes occur
across participants, so that within-band separation cannot come from the clock. It also reports the
AUC conditioned on elapsed second (mean over seconds that contain both classes in the held-out set).

Usage:
  python3 time_matched_eval.py --registry runs_registry.csv --run_id ID [--run_id ID ...] \
      [--elapsed_preds elapsed_baseline_preds.csv] [--min_prev 0.2] --out time_matched.csv
Binary tasks only (detection v4, recovery v5_binary). For the band, prevalence is computed from y_true
pooled over all runs given (they share test participants), so runs are compared on the same seconds.
"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from registry_to_seconds import load_run

def metrics(df):
    if df.y_true.nunique() < 2: return np.nan, np.nan
    return f1_score(df.y_true, df.y_pred, average="macro"), roc_auc_score(df.y_true, df.p1 if "p1" in df else df.p0)

def cond_auc(df):
    vals = []
    for s, g in df.groupby("second"):
        if g.y_true.nunique() == 2 and len(g) >= 4: vals.append(roc_auc_score(g.y_true, g.p1 if "p1" in g else g.p0))
    return np.mean(vals) if vals else np.nan, len(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True); ap.add_argument("--run_id", action="append", required=True)
    ap.add_argument("--elapsed_preds"); ap.add_argument("--min_prev", type=float, default=0.2)
    ap.add_argument("--out", default="time_matched.csv")
    a = ap.parse_args()
    runs = {rid: load_run(a.registry, rid) for rid in a.run_id}
    if a.elapsed_preds:
        e = pd.read_csv(a.elapsed_preds); e["p1"] = e.y_proba.map(lambda s: __import__("json").loads(s)[1]); runs["elapsed_time_only"] = e
    ref = runs[a.run_id[0]]
    prev = ref.groupby("second").y_true.mean()
    band = prev[(prev >= a.min_prev) & (prev <= 1 - a.min_prev)].index
    lo, hi = int(band.min()), int(band.max())
    print(f"time-matched band: seconds {lo}-{hi} (prevalence of class 1 in [{a.min_prev},{1-a.min_prev}]) -> {len(band)} seconds")
    rows = []
    for rid, df in runs.items():
        f_all, a_all = metrics(df); inb = df[df.second.isin(band)]; f_b, a_b = metrics(inb); ca, ns = cond_auc(df)
        fold_f1 = [metrics(g)[0] for _, g in inb.groupby("fold")]
        rows.append(dict(run_id=rid, F1_all=f_all, AUC_all=a_all, F1_band=f_b, AUC_band=a_b, F1_band_fold_std=np.nanstd(fold_f1, ddof=1),
                         AUC_conditioned_on_second=ca, n_seconds_used=ns, band=f"{lo}-{hi}", n_rows_band=len(inb)))
    R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(a.out, index=False)

if __name__ == "__main__":
    main()
