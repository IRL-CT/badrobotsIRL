#!/usr/bin/env python3
"""Time-matched evaluation (§5.4, Fig. 4 right panel).

In a fixed protocol the label is nearly a function of elapsed time: the
elapsed-time-only baseline reaches F1 0.976 / AUC 0.999 on v4, beating every
model in the paper. This script recomputes second-level macro-F1 and AUC

  (a) on all seconds, and
  (b) only inside the elapsed-time band where both classes occur across
      participants, so that within-band separation cannot come from the clock,

and reports the AUC conditioned on elapsed second (mean over seconds that hold
both classes in the held-out set) -- the strictest version, in which the clock
is held exactly fixed and carries no information at all by construction.

Usage:
  python3 time_matched_eval.py --registry headline_predictions.csv --run_id ID [--run_id ID ...] \
      [--elapsed_preds elapsed_baseline_preds.csv] [--min_prev 0.2] --out time_matched.csv

Handles binary (detection v4, recovery v5_binary) and 3-class (v5_multiclass)
runs; AUC is one-vs-rest macro when there are more than two classes. Give one
LABEL SCHEME per invocation: the band is computed from the first --run_id and
applied to the rest, which is only meaningful for runs that share a scheme and
test participants.
"""
import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from registry_to_seconds import load_run

def _pcols(df):
    return [c for c in df.columns if len(c) > 1 and c[0] == "p" and c[1:].isdigit()]

def _auc(yt, P):
    """Macro AUC, binary or one-vs-rest, tolerant of classes missing from a slice."""
    present = np.unique(yt)
    if len(present) < 2: return np.nan
    if P.shape[1] <= 2:
        s = P[:, 1] if P.shape[1] == 2 else P[:, 0]
        return roc_auc_score(yt, s)
    # OvR over only the classes actually present, renormalised, so a band that
    # happens to exclude one class still yields a number instead of an exception.
    aucs = [roc_auc_score((yt == c).astype(int), P[:, c]) for c in present]
    return float(np.mean(aucs))

def metrics(df):
    if df.y_true.nunique() < 2: return np.nan, np.nan
    P = df[_pcols(df)].to_numpy(float)
    return f1_score(df.y_true, df.y_pred, average="macro"), _auc(df.y_true.to_numpy(), P)

def cond_auc(df):
    vals = []
    for _, g in df.groupby("second"):
        if g.y_true.nunique() >= 2 and len(g) >= 4:
            v = _auc(g.y_true.to_numpy(), g[_pcols(g)].to_numpy(float))
            if not np.isnan(v): vals.append(v)
    return (np.mean(vals) if vals else np.nan), len(vals)

def band_seconds(ref, min_prev):
    """Seconds in which the label is genuinely mixed across participants.

    Binary: prevalence of class 1 within [min_prev, 1-min_prev].
    Multiclass: >=2 classes present and no class taking more than 1-min_prev of
    the second, which reduces to the binary rule when there are two classes.
    """
    keep = []
    for s, g in ref.groupby("second"):
        share = g.y_true.value_counts(normalize=True)
        if len(share) >= 2 and share.max() <= 1 - min_prev: keep.append(s)
    return pd.Index(sorted(keep))

def boot_ci(df, band, n_boot, seed=0):
    """Participant bootstrap for the two time-controlled numbers.

    Resamples PARTICIPANTS with replacement, not rows: the second-rows within a
    participant are serially dependent, so a row bootstrap would understate the
    interval badly. With ~24 participants this is the honest width.
    """
    rng = np.random.default_rng(seed)
    parts = df.session.unique()
    inb_all = df[df.second.isin(band)]
    ab, ac = [], []
    for _ in range(n_boot):
        pick = rng.choice(parts, size=len(parts), replace=True)
        # rebuild with repeats, tagging duplicates so groupby("second") keeps them distinct
        reps = [df[df.session == p].assign(session=f"{p}__{i}") for i, p in enumerate(pick)]
        bs = pd.concat(reps, ignore_index=True)
        v = metrics(bs[bs.second.isin(band)])[1]
        if not np.isnan(v): ab.append(v)
        v2 = cond_auc(bs)[0]
        if not np.isnan(v2): ac.append(v2)
    q = lambda v: (np.nan, np.nan) if len(v) < 2 else (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(ab), q(ac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True); ap.add_argument("--run_id", action="append", required=True)
    ap.add_argument("--elapsed_preds"); ap.add_argument("--min_prev", type=float, default=0.2)
    ap.add_argument("--out", default="time_matched.csv")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="participant-bootstrap resamples for AUC_band / AUC_conditioned CIs (0 = off)")
    a = ap.parse_args()
    runs = {rid: load_run(a.registry, rid) for rid in a.run_id}
    if a.elapsed_preds:
        e = pd.read_csv(a.elapsed_preds)
        P = np.array([json.loads(s) for s in e.y_proba], dtype=float)
        for c in range(P.shape[1]): e[f"p{c}"] = P[:, c]
        runs["elapsed_time_only"] = e
    ref = runs[a.run_id[0]]
    band = band_seconds(ref, a.min_prev)
    if not len(band): raise SystemExit(f"no second has both classes at prevalence >= {a.min_prev}")
    lo, hi = int(band.min()), int(band.max())
    print(f"time-matched band: {len(band)} seconds spanning {lo}-{hi} "
          f"(no class above {1 - a.min_prev:.2f} of the second), from {a.run_id[0]}")
    rows = []
    for rid, df in runs.items():
        f_all, a_all = metrics(df); inb = df[df.second.isin(band)]
        f_b, a_b = metrics(inb); ca, ns = cond_auc(df)
        fold_f1 = [metrics(g)[0] for _, g in inb.groupby("fold")] if "fold" in inb else []
        row = dict(run_id=rid, F1_all=f_all, AUC_all=a_all, F1_band=f_b, AUC_band=a_b,
                   F1_band_fold_std=(np.nanstd(fold_f1, ddof=1) if len(fold_f1) > 1 else np.nan),
                   AUC_conditioned_on_second=ca, n_seconds_used=ns, band=f"{lo}-{hi}",
                   n_band_seconds=len(band), n_rows_all=len(df), n_rows_band=len(inb))
        if a.bootstrap:
            (bl, bh), (cl, ch) = boot_ci(df, band, a.bootstrap)
            row.update(AUC_band_lo=bl, AUC_band_hi=bh,
                       AUC_cond_lo=cl, AUC_cond_hi=ch)
            print(f"  {rid}: AUC_band {a_b:.3f} [{bl:.3f}, {bh:.3f}] | "
                  f"AUC_cond {ca:.3f} [{cl:.3f}, {ch:.3f}]", flush=True)
        rows.append(row)
    R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(a.out, index=False)
    print(f"-> {a.out}")

if __name__ == "__main__":
    main()
