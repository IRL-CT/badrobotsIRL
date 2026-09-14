#!/usr/bin/env python3
"""Recovery decay time: after the robot acknowledges the error, how long until the
model stops calling the participant's behaviour 'error'?

For a v5_binary run (0 = error, 1 = recovery): per participant, the first second
after the recovery block starts at which P(recovery) > 0.5 for --k consecutive
seconds. Reports median, IQR, and how many participants never flip inside the
recovery block, plus the mean P(recovery) trajectory aligned to the
acknowledgment.

Two corrections over the version in corrections/:

  * it imports the FIXED `registry_to_seconds`, whose `second` is elapsed time
    within a session. The old one used frame_idx // 100 on an index into the
    concatenated all-participant dataframe, which put p5nodbot's session start
    at "second 599" and made every per-second quantity meaningless.
  * it de-duplicates (session, second). `create_data_splits` does not produce a
    clean partition -- p19nodbot lands in TWO test folds (0 and 4), giving it 58
    rows for 29 seconds. Pooling those interleaves two different models'
    predictions for the same second, which corrupts a consecutive-seconds rule
    like this one. The earlier fold is kept and the collision is reported.

Usage: python3 recovery_decay.py --registry headline_predictions.csv --run_id ID [--k 3] --out decay.csv
"""
import argparse

import numpy as np
import pandas as pd

from registry_to_seconds import load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="decay.csv")
    a = ap.parse_args()

    df = load_run(a.registry, a.run_id)
    n_before = len(df)
    df = (df.sort_values(["session", "fold", "second"])
            .drop_duplicates(["session", "second"], keep="first")
            .sort_values(["session", "second"]))
    if n_before != len(df):
        print(f"dropped {n_before - len(df)} duplicate (session, second) rows "
              f"from participants appearing in more than one test fold")

    if "p1" not in df.columns:
        raise SystemExit("need a binary run with a p1 column (v5_binary)")

    # Sanity: in v5_binary, class 1 must be the LATER block (recovery follows the
    # error). If that is reversed the whole decay reading is backwards.
    mid0 = df[df.y_true == 0].second.median()
    mid1 = df[df.y_true == 1].second.median()
    print(f"median second of class 0 = {mid0:.1f}, class 1 = {mid1:.1f} -> "
          f"class 1 is the {'later' if mid1 > mid0 else 'EARLIER'} block")
    if mid1 <= mid0:
        raise SystemExit("class 1 is not the later block; check the label scheme")

    rows, traj = [], []
    for s, g in df.groupby("session"):
        rec = g[g.y_true == 1].sort_values("second")
        if not len(rec):
            continue
        p = rec.p1.to_numpy()
        above = p > 0.5
        t_flip = np.nan
        for i in range(len(above) - a.k + 1):
            if above[i:i + a.k].all():
                t_flip = i
                break
        rows.append(dict(session=s, recovery_len_s=len(rec),
                         seconds_to_flip=t_flip,
                         frac_called_error=float((~above).mean()),
                         mean_p_recovery=float(p.mean())))
        for i, pv in enumerate(p):
            traj.append(dict(session=s, t_since_ack=i, p_recovery=pv))

    R = pd.DataFrame(rows)
    T = pd.DataFrame(traj)
    print(R.round(3).to_string(index=False))
    fl = R.seconds_to_flip
    print(f"\nflip within recovery block: {int(fl.notna().sum())}/{len(R)} participants")
    if fl.notna().any():
        print(f"median {fl.median():.1f} s (IQR {fl.quantile(.25):.1f}-{fl.quantile(.75):.1f})")
    print(f"mean fraction of recovery seconds still called error: "
          f"{R.frac_called_error.mean():.2f}")
    R.to_csv(a.out, index=False)
    T.groupby("t_since_ack").p_recovery.agg(["mean", "std", "count"]).to_csv(
        a.out.replace(".csv", "_trajectory.csv"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
