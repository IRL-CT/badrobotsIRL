#!/usr/bin/env python3
"""Per-participant second-level macro-F1 and paired tests between two runs.

This is what the abstract's "for many participants" rests on, so the count it
reports is the claim: in how many of the 24 participants does A actually beat B.

Two corrections over the version in corrections/:

  * it imports the FIXED `registry_to_seconds` (elapsed time within a session,
    not an index into the concatenated all-participant dataframe);
  * it de-duplicates (session, second). p19nodbot appears in TWO test folds, so
    pooling gives it 58 rows for 29 seconds and double weight in a
    "participants where A > B" count. The earlier fold is kept.

On the statistics: participants inside a fold share a model, so the 24
per-participant scores are not independent. The per-fold test (n=5) is the
conservative one; both are reported and the paper should say which it quotes.
With n=5 the Wilcoxon cannot go below p=0.0625 however large the effect, so a
non-significant fold-level result is not evidence of no effect.

Usage:
  python3 per_participant_and_paired.py --registry headline_predictions.csv \
      --run_a ID_A --run_b ID_B [--label_a Gemini --label_b engineered] \
      --out per_participant.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from registry_to_seconds import load_run


def dedupe(df):
    return (df.sort_values(["session", "fold", "second"])
              .drop_duplicates(["session", "second"], keep="first"))


def per_part(df):
    out = {}
    for s, g in df.groupby("session"):
        out[s] = (f1_score(g.y_true, g.y_pred, average="macro")
                  if g.y_true.nunique() > 1 else np.nan)
    return pd.Series(out)


def per_fold(df):
    out = {}
    for k, g in df.groupby("fold"):
        out[k] = f1_score(g.y_true, g.y_pred, average="macro")
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--out", default="per_participant.csv")
    a = ap.parse_args()

    A, B = dedupe(load_run(a.registry, a.run_a)), dedupe(load_run(a.registry, a.run_b))
    # Per-fold F1 must come from the UN-deduped frame or fold 4 loses p19nodbot's
    # rows to fold 0; per-participant must come from the deduped one.
    fa, fb = per_fold(load_run(a.registry, a.run_a)), per_fold(load_run(a.registry, a.run_b))
    pa, pb = per_part(A), per_part(B)

    df = pd.DataFrame({a.label_a: pa, a.label_b: pb})
    df["diff"] = df[a.label_a] - df[a.label_b]
    print(df.round(3).to_string())
    df.to_csv(a.out)

    d = df.dropna()
    print(f"\nper-participant ({len(d)} participants with both classes present):")
    print(f"  {a.label_a}: mean {d[a.label_a].mean():.3f} (sd {d[a.label_a].std():.3f})")
    print(f"  {a.label_b}: mean {d[a.label_b].mean():.3f} (sd {d[a.label_b].std():.3f})")
    n_win = int((d["diff"] > 0).sum())
    print(f"  {a.label_a} > {a.label_b} in {n_win}/{len(d)} participants "
          f"({100 * n_win / len(d):.0f}%)")
    if d[a.label_a].nunique() > 1 or d[a.label_b].nunique() > 1:
        print(f"  Wilcoxon p={stats.wilcoxon(d[a.label_a], d[a.label_b]).pvalue:.4f}; "
              f"paired t p={stats.ttest_rel(d[a.label_a], d[a.label_b]).pvalue:.4f}")
        # sign test: how surprising is n_win out of n, under a fair coin
        pb_ = stats.binomtest(n_win, len(d), 0.5).pvalue
        print(f"  sign test on the participant count: p={pb_:.4f}")

    print(f"\nper-fold (n={len(fa)}, the conservative test -- participants inside a "
          f"fold share a model):")
    print(f"  {a.label_a}: {[round(x, 3) for x in fa.tolist()]}")
    print(f"  {a.label_b}: {[round(x, 3) for x in fb.tolist()]}")
    print(f"  Wilcoxon p={stats.wilcoxon(fa, fb).pvalue:.4f}; "
          f"paired t p={stats.ttest_rel(fa, fb).pvalue:.4f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
