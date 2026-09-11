#!/usr/bin/env python3
"""Recovery decay time (§5.3): after the robot's acknowledgment, how long until the model stops calling
the participant's behaviour 'error'?

For a v5_binary run (0 = error, 1 = recovery): per participant, the first second t after the recovery
block starts such that P(recovery) > 0.5 for --k consecutive seconds. Reports median, IQR, and the
number of participants for whom this never happens within the recovery block. Also outputs the mean
P(recovery) trajectory aligned to the acknowledgment (for Fig. 1 band C / Supp E).

Usage: python3 recovery_decay.py --registry runs_registry.csv --run_id ID [--k 3] --out decay.csv
"""
import argparse, numpy as np, pandas as pd
from registry_to_seconds import load_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True); ap.add_argument("--run_id", required=True)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--out", default="decay.csv")
    a = ap.parse_args()
    df = load_run(a.registry, a.run_id).sort_values(["session", "second"])
    rows, traj = [], []
    for s, g in df.groupby("session"):
        rec = g[g.y_true == 1]
        if not len(rec): continue
        t0 = rec.second.min(); p = rec.p1.to_numpy(); above = p > 0.5
        t_flip = np.nan
        for i in range(len(above) - a.k + 1):
            if above[i:i + a.k].all(): t_flip = i; break
        rows.append(dict(session=s, recovery_len_s=len(rec), seconds_to_flip=t_flip, frac_called_error=float((~above).mean())))
        for i, pv in enumerate(p): traj.append(dict(session=s, t_since_ack=i, p_recovery=pv))
    R = pd.DataFrame(rows); T = pd.DataFrame(traj)
    print(R.to_string(index=False))
    fl = R.seconds_to_flip
    print(f"\nflip within recovery block: {fl.notna().sum()}/{len(R)} participants; median {fl.median():.1f} s (IQR {fl.quantile(.25):.1f}-{fl.quantile(.75):.1f}); "
          f"mean fraction of recovery seconds still called error: {R.frac_called_error.mean():.2f}")
    R.to_csv(a.out, index=False); T.groupby("t_since_ack").p_recovery.agg(["mean", "std", "count"]).to_csv(a.out.replace(".csv", "_trajectory.csv"))

if __name__ == "__main__":
    main()
