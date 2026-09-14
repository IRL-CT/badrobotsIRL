#!/usr/bin/env python3
"""Second-level 3-class confusion for v5 recovery runs (NEUTRAL / ERROR / RECOVERY).

Label mapping is recovery_labels.py's: multiclass_label 0 = NEUTRAL,
1 = ERROR, 2 = RECOVERY.

The row that matters for the paper is RECOVERY: a model that calls recovery
"error" keeps a robot apologising after the person has already moved on, and
that specific cell (true RECOVERY -> predicted ERROR) is reported explicitly
rather than being folded into a macro average.

De-duplicates (session, second): p19nodbot appears in two test folds, so
pooling would count it twice and mix two models' predictions.

Usage:
  python3 confusion_3class.py --registry headline_predictions.csv \
      --run_id ID [--run_id ID ...] --out confusion_3class.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from registry_to_seconds import load_run

NAMES = {0: "NEUTRAL", 1: "ERROR", 2: "RECOVERY"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--run_id", action="append", required=True)
    ap.add_argument("--out", default="confusion_3class.csv")
    a = ap.parse_args()

    rows = []
    for rid in a.run_id:
        df = load_run(a.registry, rid)
        n0 = len(df)
        df = df.sort_values(["session", "fold", "second"]).drop_duplicates(
            ["session", "second"], keep="first")
        yt = df.y_true.to_numpy().astype(int)
        yp = df.y_pred.to_numpy().astype(int)
        labs = [0, 1, 2]
        C = confusion_matrix(yt, yp, labels=labs)
        Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)

        print(f"\n=== {rid} ===")
        print(f"{n0 - len(df)} duplicate (session, second) rows dropped; "
              f"{len(df)} second-level rows, {df.session.nunique()} participants")
        print("\nconfusion (rows = true, cols = predicted), counts:")
        hdr = "".join(f"{NAMES[c]:>10s}" for c in labs)
        print(f"{'':>10s}{hdr}")
        for i, c in enumerate(labs):
            print(f"{NAMES[c]:>10s}" + "".join(f"{C[i, j]:10d}" for j in range(3)))
        print("\nrow-normalised (recall per true class):")
        print(f"{'':>10s}{hdr}")
        for i, c in enumerate(labs):
            print(f"{NAMES[c]:>10s}" + "".join(f"{Cn[i, j]:10.3f}" for j in range(3)))
        print("\nper class:")
        for i, c in enumerate(labs):
            print(f"  {NAMES[c]:9s} precision {p[i]:.3f}  recall {r[i]:.3f}  "
                  f"F1 {f[i]:.3f}  support {s[i]}")
        print(f"  macro-F1 {f.mean():.3f}")
        rec_as_err = Cn[2, 1]
        err_as_rec = Cn[1, 2]
        print(f"\n  true RECOVERY called ERROR:  {rec_as_err:.3f} "
              f"({C[2, 1]} of {s[2]} seconds)")
        print(f"  true ERROR called RECOVERY:  {err_as_rec:.3f} "
              f"({C[1, 2]} of {s[1]} seconds)")

        row = dict(run_id=rid, n_seconds=len(df), n_participants=df.session.nunique(),
                   macro_F1=round(float(f.mean()), 3))
        for i, c in enumerate(labs):
            row[f"precision_{NAMES[c]}"] = round(float(p[i]), 3)
            row[f"recall_{NAMES[c]}"] = round(float(r[i]), 3)
            row[f"F1_{NAMES[c]}"] = round(float(f[i]), 3)
            row[f"support_{NAMES[c]}"] = int(s[i])
        for i, c in enumerate(labs):
            for j, c2 in enumerate(labs):
                row[f"n_true{NAMES[c]}_pred{NAMES[c2]}"] = int(C[i, j])
                row[f"frac_true{NAMES[c]}_pred{NAMES[c2]}"] = round(float(Cn[i, j]), 3)
        rows.append(row)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
