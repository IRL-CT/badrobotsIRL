#!/usr/bin/env python3
"""Section A noise-input control for the recurrent hierarchy, resumable.

Grid (72 runs): level2 x level1 x window_sec x noise_input x seed
                {none,gru,lstm} x {mean,gru} x {1,3} x {False,True} x {0,1,2}

The decision rule from hier_noise_control_patch.md: if noise-input runs with
level2 != "none" reach second-level F1 >= 0.85, the recurrent hierarchy is
reported only as a demonstration of the protocol confound, because a model fed
pure noise would be scoring off the session structure alone.

Two things this fixes about the first attempt:
  * It ran as a login-node nohup and was killed when the session ended, 25/72 in
    (and all 47 survivors were the level2 != none runs -- i.e. every run the
    decision actually depends on). This is a batch job now.
  * It only wrote its CSV at the very end, so nine hours of compute produced
    nothing. Results are appended after every run.

Resume is automatic: completed (level1, level2, window_sec, noise, seed) keys are
read back from the output CSV and skipped.
"""

import argparse
import itertools
import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "common")))

OUT = os.path.join(HERE, "noise_control_hier.csv")
COLS = ["level1", "level2", "window_sec", "noise", "seed", "F1", "AUC"]

BASE = dict(binary_multiclass="binary", units_l1=32, units_l2=32, stride_sec=1,
            dropout=0.4, l2_reg=1e-4, label_smoothing=0.05, optimizer="adam",
            learning_rate=1e-3, batch_size=4, epochs=60, patience=10,
            aux_localization=False, aux_weight=0.5, feature_set="full",
            gemini_source="visual_audio", gemini_dims=128, modality="facial")


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return {(r.level1, r.level2, int(r.window_sec), bool(r.noise), int(r.seed))
            for r in d.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    logs = {}
    stub = types.ModuleType("wandb")
    stub.config = None
    stub.init = lambda *x, **k: None
    stub.log = lambda d: logs.update(d)
    sys.modules["wandb"] = stub
    import hier_model as HM

    already = done_keys(a.out)
    grid = list(itertools.product(["none", "gru", "lstm"], ["mean", "gru"],
                                  [1, 3], [False, True], [0, 1, 2]))
    todo = [g for g in grid
            if (g[1], g[0], g[2], g[3], g[4]) not in already]
    print(f"grid {len(grid)} | already done {len(already)} | to run {len(todo)}",
          flush=True)

    for i, (l2, l1, ws, noise, seed) in enumerate(todo, 1):
        cfg = dict(BASE)
        cfg.update(level1=l1, level2=l2, window_sec=ws,
                   noise_input=noise, seed=seed)
        stub.config = Cfg(cfg)
        logs.clear()
        try:
            HM.train()
        except Exception as e:
            print(f"[{i}/{len(todo)}] ERR {l1}/{l2}/w{ws}/"
                  f"{'noise' if noise else 'real'}/s{seed}: "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        row = dict(level1=l1, level2=l2, window_sec=ws, noise=noise, seed=seed,
                   F1=logs.get("avg_test_f1", np.nan),
                   AUC=logs.get("avg_test_auc", np.nan))
        # Append immediately: a nine-hour job must not lose everything if killed.
        pd.DataFrame([row])[COLS].to_csv(
            a.out, mode="a", header=not os.path.exists(a.out), index=False)
        print(f"[{i}/{len(todo)}] {l1}/{l2}/w{ws}/"
              f"{'noise' if noise else 'real'}/s{seed} "
              f"F1={row['F1']:.3f} AUC={row['AUC']:.3f}", flush=True)

    d = pd.read_csv(a.out)
    print("\n=== SECTION A: noise-input control ===")
    print(d.groupby(["level2", "noise"])[["F1", "AUC"]]
           .agg(["count", "median", "max"]).round(3))
    nz = d[(d.noise) & (d.level2 != "none")]
    if len(nz):
        print(f"\nDECISION RULE (noise, level2 != none, F1 >= 0.85): "
              f"{int((nz.F1 >= 0.85).sum())} of {len(nz)} | max {nz.F1.max():.3f}")
        print("  -> " + ("FAILS: report v6 only as a confound demonstration"
                         if (nz.F1 >= 0.85).any() else
                         "PASSES: v6 numbers may be used"))
    else:
        print("\nno noise runs with level2 != none yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
