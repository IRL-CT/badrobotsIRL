#!/usr/bin/env python3
"""v6 (hierarchical escalation): best result per arm, with its noise-control verdict.

v6 is the one track where a leaderboard on its own would actively mislead, so no
number here is reported without the control that decides whether it means
anything.

The control replaces the input features with Gaussian noise and leaves the
session structure identical. Decision rule (hier_noise_control_patch.md): if
noise-input runs reach second-level F1 >= 0.85, that arm is reported only as a
demonstration of the protocol confound -- a model fed pure noise scoring that
high can only be reading the session schedule.

VALIDATION SOURCE. The primary evidence is the sweep's OWN `noise_input` runs
(~3k of them), not the two small dedicated grids in hierarchical_escalation/
(72 and 42 runs). Two reasons:

  * coverage -- the dedicated grids pin one configuration family (modality
    `facial`, feature_set `full`, gemini_dims 128) and vary only
    level1/level2/window_sec/seed, with window_sec in {1,3}. The sweep's best
    runs use other modalities and window_sec 5, i.e. configurations the
    dedicated control never tested. Validating a 0.808 run against a grid whose
    real runs topped out at 0.682 would not be validating that run at all.
  * completeness -- the dedicated grid never ran `level2=rnn`, a real arm with
    563 runs in the sweep. It fails the control decisively (noise max 0.986 vs
    real max 0.979), so omitting it would have left an unvalidated arm sitting
    at the top of v6.

The dedicated grids are still reported alongside as a cross-check; they agree
with the sweep on every arm they share.

Measured outcome:

  * flat 1 Hz (level2=none)  -- PASSES. real max 0.808 vs noise max 0.560.
  * escalation features      -- PASSES. real max 0.961 vs noise max 0.703.
  * session GRU / LSTM / RNN -- FAIL. Noise reaches 0.967 / 0.974 / 0.986, level
    with (RNN: above) the best real run of the same arm. With level-2 recurrence
    the hidden state can count windows, and in a fixed protocol the count
    determines the label.

Usage
-----
    python3 make_v6_summary.py --raw analysis_results/raw --out analysis_results
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HIER = os.path.normpath(os.path.join(HERE, "..", "hierarchical_escalation"))
THRESH = 0.85

KEEP = ["binary_multiclass", "level1", "level2", "model_type", "modality", "feature_set",
        "gemini_source", "gemini_dims", "window_sec", "stride_sec",
        "units_l1", "units_l2", "dropout", "l2_reg", "learning_rate", "batch_size",
        "epochs", "patience", "label_smoothing", "optimizer", "aux_localization",
        "aux_weight", "use_current", "use_delta_base", "use_delta_prev", "use_drift",
        "noise_input", "seed"]


def load_sweep(raw):
    p = os.path.join(raw, "brirl_hier_escalation_v6.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    W = pd.read_csv(p, low_memory=False)
    W = W[W.State == "finished"].copy()
    W["f1"] = pd.to_numeric(W.get("avg_sec_test_f1"), errors="coerce")
    W = W.dropna(subset=["f1"])
    if "noise_input" in W.columns:
        W["noise"] = W["noise_input"].apply(lambda v: v is True or str(v).lower() == "true")
    else:
        W["noise"] = False
    W["arm"] = np.where(
        W.level2.notna(),
        "session " + W.level2.astype(str).str.upper() + " (level2=" + W.level2.astype(str) + ")",
        "escalation features (per-window + escalation inputs)")
    W.loc[W.level2 == "none", "arm"] = "flat 1 Hz (level2=none)"
    return W


def control_grids():
    """The two dedicated noise-control grids, kept as a cross-check."""
    out = {}
    pA = os.path.join(HIER, "noise_control_hier.csv")
    if os.path.exists(pA):
        A = pd.read_csv(pA)
        A["noise"] = A["noise"].astype(bool)
        for l2 in A.level2.unique():
            k = ("flat 1 Hz (level2=none)" if l2 == "none"
                 else f"session {str(l2).upper()} (level2={l2})")
            s = A[A.level2 == l2]
            out[k] = (s[~s.noise].F1, s[s.noise].F1)
    pB = os.path.join(HIER, "noise_control_escalation.csv")
    if os.path.exists(pB):
        B = pd.read_csv(pB)
        B["noise"] = B["noise"].astype(bool)
        out["escalation features (per-window + escalation inputs)"] = (
            B[~B.noise].F1, B[B.noise].F1)
    return out


def mm(v):
    v = pd.to_numeric(v, errors="coerce").dropna()
    return (len(v), round(float(v.median()), 3), round(float(v.max()), 3)) if len(v) else (0, "", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="analysis_results/raw")
    ap.add_argument("--out", default="analysis_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    W = load_sweep(a.raw)
    grids = control_grids()

    summary, best = [], []
    for arm, sub in W.groupby("arm"):
        real, noise = sub[~sub.noise], sub[sub.noise]
        rn, rmed, rmax = mm(real.f1)
        nn, nmed, nmax = mm(noise.f1)
        over = int((pd.to_numeric(noise.f1, errors="coerce") >= THRESH).sum()) if nn else ""
        if not nn:
            status = "NOT VALIDATED (no noise runs)"
            why = "no noise-input run exists for this arm"
        elif over:
            status = "FAILED noise control"
            why = (f"{over} noise run(s) reach F1 >= {THRESH}; noise max {nmax} vs "
                   f"real max {rmax} -- the score is reachable with no real signal")
        else:
            status = "VALIDATED"
            why = (f"noise max {nmax} stays below {THRESH}; real max {rmax} "
                   f"exceeds it by {round(rmax - nmax, 3)}")

        gr = grids.get(arm)
        g_real = g_noise = ""
        if gr is not None:
            g_real = mm(gr[0])[2]
            g_noise = mm(gr[1])[2]

        summary.append(dict(
            arm=arm, validation_status=status,
            real_n=rn, real_median_F1=rmed, real_best_F1=rmax,
            noise_n=nn, noise_median_F1=nmed, noise_max_F1=nmax,
            real_max_minus_noise_max=(round(rmax - nmax, 3) if nn else ""),
            noise_runs_at_or_above_threshold=over, threshold=THRESH,
            dedicated_grid_real_max=g_real, dedicated_grid_noise_max=g_noise,
            reason=why))

        if len(real):
            r = real.loc[real.f1.idxmax()]

            def g(k):
                if k not in real.columns:
                    return ""
                v = r.get(k)
                return "" if pd.isna(v) else v

            row = dict(arm=arm, validation_status=status,
                       F1=round(float(r.f1), 3),
                       F1_ci_low=g("avg_sec_test_f1_ci_low"),
                       F1_ci_high=g("avg_sec_test_f1_ci_high"),
                       AUC=g("avg_sec_test_auc"), FNR=g("avg_sec_test_fnr"),
                       Accuracy=g("avg_sec_test_accuracy"),
                       Precision=g("avg_sec_test_precision"),
                       Recall=g("avg_sec_test_recall"),
                       n_real_runs_in_arm=len(real),
                       median_F1_in_arm=round(float(real.f1.median()), 3))
            for k in KEEP:
                row[k] = g(k)
            row.update(run_name=g("Name"), run_id=g("ID"), sweep=g("Sweep"),
                       run_url=g("URL"), created=g("Created"),
                       usable_in_paper=("yes" if status == "VALIDATED" else "NO"))
            best.append(row)

    S = pd.DataFrame(summary).sort_values(["validation_status", "arm"])
    B = pd.DataFrame(best).sort_values(["validation_status", "F1"], ascending=[True, False])
    p1 = os.path.join(a.out, "v6_validated_summary.csv")
    p2 = os.path.join(a.out, "v6_best_runs_by_arm.csv")
    S.to_csv(p1, index=False)
    B.to_csv(p2, index=False)

    pd.set_option("display.width", 250)
    print(S[["arm", "validation_status", "real_n", "real_best_F1", "noise_n",
             "noise_max_F1", "real_max_minus_noise_max",
             "dedicated_grid_noise_max"]].to_string(index=False))
    print(f"\n-> {p1}\n-> {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
