#!/usr/bin/env python3
"""Shared loader: turn runs_registry.csv rows for one run_id into one row per (participant, second).

Collapses the model's evaluated rows to 1 Hz: majority vote for y_pred, mean for y_proba, label of the
last frame for y_true (same convention as eval_utils.aggregate_blocks). Requires the registry patch
(frame_idx, y_proba columns).
"""
import json, numpy as np, pandas as pd

def load_run(registry_csv, run_id):
    reg = pd.read_csv(registry_csv, dtype=str); reg = reg[reg.run_id == run_id]
    if not len(reg): raise SystemExit(f"run_id {run_id} not in {registry_csv}")
    out = []
    for _, r in reg.iterrows():
        yp = np.array(json.loads(r.y_pred)); yt = np.array(json.loads(r.y_true)); ss = np.array(json.loads(r.session))
        fi = np.array(json.loads(r.frame_idx)); pr = np.array(json.loads(r.y_proba), dtype=float)
        if pr.ndim == 1: pr = pr[:, None]
        sec = fi // 100                                           # base grid is 100 fps
        df = pd.DataFrame(dict(session=ss, second=sec, y_pred=yp, y_true=yt))
        for c in range(pr.shape[1]): df[f"p{c}"] = pr[:, c]
        g = df.groupby(["session", "second"])
        agg = g.agg(y_pred=("y_pred", lambda x: x.value_counts().idxmax()), y_true=("y_true", "last"),
                    **{f"p{c}": (f"p{c}", "mean") for c in range(pr.shape[1])}).reset_index()
        agg["fold"] = int(r.fold); out.append(agg)
    return pd.concat(out, ignore_index=True)
