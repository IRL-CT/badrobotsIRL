#!/usr/bin/env python3
"""Per-participant second-level F1 (Fig. 5 / Supp E) and paired tests between two runs (§5.1, §5.3).

Usage:
  python3 per_participant_and_paired.py --registry runs_registry.csv --run_a ID_A --run_b ID_B \
      [--label_a "Gemini" --label_b "conventional"] --out per_participant.csv
Reports: per-participant F1 for both runs; paired Wilcoxon and t-test over the 5 folds and over the 24
participants (participants within a fold share a model — say so in the paper); number of participants
where A > B.
"""
import argparse, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from registry_to_seconds import load_run

def per_part(df):
    return df.groupby("session").apply(lambda g: f1_score(g.y_true, g.y_pred, average="macro") if g.y_true.nunique() > 1 else np.nan)
def per_fold(df):
    return df.groupby("fold").apply(lambda g: f1_score(g.y_true, g.y_pred, average="macro"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True); ap.add_argument("--run_a", required=True); ap.add_argument("--run_b", required=True)
    ap.add_argument("--label_a", default="A"); ap.add_argument("--label_b", default="B"); ap.add_argument("--out", default="per_participant.csv")
    a = ap.parse_args()
    A, B = load_run(a.registry, a.run_a), load_run(a.registry, a.run_b)
    pa, pb = per_part(A), per_part(B); fa, fb = per_fold(A), per_fold(B)
    df = pd.DataFrame({a.label_a: pa, a.label_b: pb}); df["diff"] = df[a.label_a] - df[a.label_b]
    print(df.round(3).to_string()); df.to_csv(a.out)
    d = df.dropna()
    print(f"\nper-participant: {a.label_a} mean {d[a.label_a].mean():.3f} (sd {d[a.label_a].std():.3f}); {a.label_b} mean {d[a.label_b].mean():.3f} (sd {d[a.label_b].std():.3f})")
    print(f"A>B in {(d['diff']>0).sum()}/{len(d)} participants; Wilcoxon p={stats.wilcoxon(d[a.label_a], d[a.label_b]).pvalue:.3f}; paired t p={stats.ttest_rel(d[a.label_a], d[a.label_b]).pvalue:.3f}")
    print(f"per-fold: A {fa.round(3).tolist()} B {fb.round(3).tolist()}; Wilcoxon p={stats.wilcoxon(fa, fb).pvalue:.3f}; paired t p={stats.ttest_rel(fa, fb).pvalue:.3f}")

if __name__ == "__main__":
    main()
