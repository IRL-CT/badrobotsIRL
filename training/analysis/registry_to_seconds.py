#!/usr/bin/env python3
"""Shared loader: turn runs_registry.csv rows for one run_id into one row per (participant, second).

Collapses the model's evaluated rows to 1 Hz: majority vote for y_pred, mean for y_proba, label of the
last frame for y_true (same convention as eval_utils.aggregate_blocks). Requires the registry patch
(frame_idx, y_proba columns).

`second` is seconds SINCE THAT PARTICIPANT'S SESSION START, which is what the
time-matched analysis and `elapsed_time_baseline.py` (np.arange per participant)
both mean by elapsed time. It is NOT frame_idx // 100: frame_idx indexes the
concatenated all-participant dataframe, so each participant sits in its own
contiguous global block (p18nodbot at 18401, p5nodbot at 59981) and the raw
quotient would report p5's session start as "second 599". Subtracting the
participant's own first row fixes this. The folds are participant-disjoint, so a
participant that appears in a test split has ALL of its rows there and that
minimum really is its first windowed row -- which trails true session start by a
uniform (window - 1) frames (< 0.3 s), the same constant for every participant,
so it shifts no comparison.
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
        df = pd.DataFrame(dict(session=ss, frame_idx=fi, y_pred=yp, y_true=yt))
        # elapsed within session, not position in the concatenated dataframe
        df["second"] = (df.frame_idx - df.groupby("session").frame_idx.transform("min")) // 100
        for c in range(pr.shape[1]): df[f"p{c}"] = pr[:, c]
        g = df.groupby(["session", "second"])
        agg = g.agg(y_pred=("y_pred", lambda x: x.value_counts().idxmax()), y_true=("y_true", "last"),
                    **{f"p{c}": (f"p{c}", "mean") for c in range(pr.shape[1])}).reset_index()
        agg["fold"] = int(r.fold); out.append(agg)
    return pd.concat(out, ignore_index=True)
