#!/usr/bin/env python3
"""Elapsed-time-only baseline (§4.6 / Fig. 4): how well does the clock alone predict the label?

Participant-disjoint 5-fold CV identical to the sweeps (hier_data.participant_folds). Features:
  --features elapsed         : seconds since session start
  --features elapsed,speaking: + a 1/0 flag for 'participant is speaking' derived from the openSMILE block
                               (nonzero acoustic features in that second); shows how much of the audio
                               signal is simply speech onset.
Labels: --scheme v4 (detection, recovery block dropped) | v5_binary (error vs recovery) | v5_multiclass.
Outputs second-level macro-F1 / AUC per fold + participant-bootstrap CI, and per-second predictions
(for time_matched_eval.py) to --out_preds.

Run from training/hierarchical_escalation/ (imports hier_data, label_utils) with the repo's data/ present.
"""
import argparse, os, sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
# These scripts live in training/analysis/, so the modules they need are SIBLINGS.
# The original join produced training/analysis/../training/hierarchical_escalation.
for _p in ("hierarchical_escalation", "recovery", "temp_allscripts", "common"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", _p)))
import hier_data as H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="v4", choices=["v4","v5_binary","v5_multiclass"])
    ap.add_argument("--features", default="elapsed")
    ap.add_argument("--out_preds", default="elapsed_baseline_preds.csv")
    a = ap.parse_args()
    # frame-level info + audio block, on the base grid
    info, X = H.load_frames("audio", feature_set="full")          # 92-col audio slice, recovery block dropped (v4)
    if a.scheme.startswith("v5"):
        from recovery_labels import apply_scheme
        base = pd.read_csv(H.BASE_CSV)
        df, mask = apply_scheme(base, "binary" if a.scheme=="v5_binary" else "multiclass")
        info = df.iloc[:, :4].reset_index(drop=True); X = df.iloc[:, 63:88].to_numpy("float32")
    Xs, yb, ym, parts = H.to_seconds(info, X)
    y = ym if a.scheme=="v5_multiclass" else yb
    # elapsed seconds within participant
    elapsed = np.concatenate([np.arange((parts==p).sum()) for p in pd.unique(parts)]).astype(float)
    feats = [elapsed[:,None]]
    if "speaking" in a.features: feats.append((np.abs(Xs).sum(1) > 1e-6).astype(float)[:,None])
    F = np.hstack(feats)
    folds = H.participant_folds(parts)
    rows, preds = [], []
    for k,(tr,va,te) in enumerate(folds):
        trm = np.isin(parts, np.concatenate([tr,va])); tem = np.isin(parts, te)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(F[trm], y[trm])
        pr = clf.predict_proba(F[tem]); yp = clf.predict(F[tem]); yt = y[tem]
        auc = roc_auc_score(yt, pr[:,1]) if pr.shape[1]==2 else roc_auc_score(yt, pr, multi_class="ovr", average="macro")
        rows.append(dict(fold=k, f1=f1_score(yt, yp, average="macro"), auc=auc, n_sec=int(tem.sum())))
        for s,e,t,p_,pp in zip(parts[tem], elapsed[tem], yt, yp, pr.tolist()):
            preds.append(dict(fold=k, session=s, second=int(e), y_true=int(t), y_pred=int(p_), y_proba=json.dumps(pp)))
    R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
    print(f"ELAPSED-TIME BASELINE [{a.scheme}, {a.features}]  F1 {R.f1.mean():.3f} ± {R.f1.std(ddof=1):.3f}   AUC {R.auc.mean():.3f} ± {R.auc.std(ddof=1):.3f}")
    pd.DataFrame(preds).to_csv(a.out_preds, index=False); print("preds ->", a.out_preds)

if __name__ == "__main__":
    main()
