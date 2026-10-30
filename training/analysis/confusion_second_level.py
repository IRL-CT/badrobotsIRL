#!/usr/bin/env python3
"""Second-level pooled confusion matrix for any run in a headline registry.

Generalises confusion_3class.py to any number of classes, so it serves the
4-class stage task (T10: survey / Error I / II / III) as well as the recovery
schemes. Predictions are collapsed to one row per (participant, second) by
registry_to_seconds.load_run (majority vote), pooled over the five test folds,
and de-duplicated on (session, second) because p19nodbot sits in two test folds.

Usage:
  python3 confusion_second_level.py --registry rebuttal/headline_predictions_rebuttal_A.csv \
      --run_id headline_11_stage_best_abs_exact --names survey,ErrorI,ErrorII,ErrorIII \
      --out rebuttal/confusion_stage_absolute.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from registry_to_seconds import load_run


def pooled_confusion(df, labels):
    yt = df.y_true.to_numpy().astype(int)
    yp = df.y_pred.to_numpy().astype(int)
    C = confusion_matrix(yt, yp, labels=labels)
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return C, Cn, p, r, f, s


def print_matrix(C, Cn, names):
    hdr = "".join(f"{n:>10s}" for n in names)
    print(f"{'':>10s}{hdr}")
    for i, n in enumerate(names):
        print(f"{n:>10s}" + "".join(f"{C[i, j]:10d}" for j in range(len(names))))
    print("row-normalised (recall per true class):")
    print(f"{'':>10s}{hdr}")
    for i, n in enumerate(names):
        print(f"{n:>10s}" + "".join(f"{Cn[i, j]:10.3f}" for j in range(len(names))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--run_id", action="append", required=True)
    ap.add_argument("--names", default=None, help="comma-separated class names, in label order")
    ap.add_argument("--out", default="confusion_second_level.csv")
    a = ap.parse_args()

    rows = []
    for rid in a.run_id:
        df = load_run(a.registry, rid)
        n0 = len(df)
        df = df.sort_values(["session", "fold", "second"]).drop_duplicates(["session", "second"], keep="first")
        labels = sorted(set(df.y_true.astype(int)) | set(df.y_pred.astype(int)))
        names = a.names.split(",") if a.names else [str(l) for l in labels]
        C, Cn, p, r, f, s = pooled_confusion(df, labels)
        print(f"\n=== {rid} ===  {n0 - len(df)} duplicate rows dropped; {len(df)} seconds, "
              f"{df.session.nunique()} participants; macro-F1 {f.mean():.3f}")
        print_matrix(C, Cn, names)
        adj = float(np.mean(np.abs(df.y_true.astype(int) - df.y_pred.astype(int)) <= 1))
        row = dict(run_id=rid, n_seconds=len(df), n_participants=df.session.nunique(),
                   macro_F1=round(float(f.mean()), 4), within_one_class_acc=round(adj, 4),
                   confusion_counts=json.dumps(C.tolist()),
                   confusion_rownorm=json.dumps(np.round(Cn, 4).tolist()))
        for i, n in enumerate(names):
            row[f"precision_{n}"] = round(float(p[i]), 4)
            row[f"recall_{n}"] = round(float(r[i]), 4)
            row[f"F1_{n}"] = round(float(f[i]), 4)
            row[f"support_{n}"] = int(s[i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
