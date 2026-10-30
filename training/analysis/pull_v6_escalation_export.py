#!/usr/bin/env python3
"""Pull the escalation (relative-feature) arm of brirl_hier_escalation_v6 from W&B.

One row per run with config + the second-level metrics. Used by the rebuttal
report for the baseline-length table (T11, as swept) and the block ladder (T1).

Two things the earlier analysis_results export handled implicitly and this one
makes explicit:
  * runs created before the `noise_input` / `seed` switches were added have
    those config keys unset; they are REAL runs (noise_input is filled as False);
  * only state == finished is kept, as make_v6_summary.py did.

    python3 pull_v6_escalation_export.py --out rebuttal/v6_escalation_export.csv
"""
import argparse
import os

import pandas as pd
import wandb

KEYS = ["binary_multiclass", "noise_input", "seed", "model_type", "use_current", "use_delta_base",
        "use_delta_prev", "use_drift", "use_scalars", "baseline_sec", "window_sec", "stride_sec",
        "feature_set", "gemini_source", "gemini_dims", "modality", "C", "alpha", "n_estimators",
        "max_depth", "units_l1"]
SUMMARY = ["avg_sec_test_f1", "avg_sec_test_auc", "avg_sec_test_fnr", "avg_sec_test_f1_ci_low",
           "avg_sec_test_f1_ci_high", "avg_test_f1", "n_features_raw", "n_features_escalation", "n_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="mteresaparreira")
    ap.add_argument("--project", default="brirl_hier_escalation_v6")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "rebuttal", "v6_escalation_export.csv"))
    a = ap.parse_args()
    api = wandb.Api(timeout=120)
    rows = []
    for r in api.runs(f"{a.entity}/{a.project}", per_page=500):
        c = r.config
        if "use_delta_base" not in c:      # recurrent (hier_model.py) arm
            continue
        s = r.summary
        row = dict(run_id=r.id, name=r.name, state=r.state, created=str(r.created_at))
        row.update({k: c.get(k) for k in KEYS})
        row.update({k: s.get(k) for k in SUMMARY})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["noise_input"] = df["noise_input"].fillna(False).astype(bool)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    fin = df[df.state == "finished"]
    print(f"{len(df)} escalation runs ({len(fin)} finished) -> {a.out}")
    print(fin.groupby(["binary_multiclass", "noise_input"]).avg_sec_test_f1.agg(["count", "max", "median"]).round(3))


if __name__ == "__main__":
    main()
