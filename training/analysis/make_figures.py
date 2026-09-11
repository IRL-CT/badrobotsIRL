#!/usr/bin/env python3
"""Paper figures from the results table, W&B exports, and analysis outputs.

  fig2  dot+CI plot of second-level F1 per input set, detection (top) and recovery (bottom)
        needs: --results results_by_track.csv
  fig3  3x3 recovery confusion matrices (best conventional | best Gemini) + per-class F1
        needs: --recovery_linear/--recovery_deep exports (t{k}_confusion_matrix / fold_{k}_confusion_matrix)
  fig4  clock confound: label-vs-time raster (left; needs --base_csv) and all-vs-band AUC bars (right; needs --time_matched)
  fig5  per-participant F1 strip plot (needs --per_participant CSV from per_participant_and_paired.py)
Usage: python3 make_figures.py fig2 --results R.csv --out figs/
"""
import argparse, json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fig2(a):
    R = pd.read_csv(a.results); R = R[(R.task == "binary") & (R.model_family == "pooled").eq(False)]
    order = ["F", "A", "P", "F+A", "P+F+A", "P+F+A+T", "P+F+A+C", "BEST conventional (any)",
             "G [visual]", "G [visual_audio]", "BEST Gemini-inclusive (any)"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True)
    for ax, track in zip(axes, ["detection", "recovery"]):
        sub = R[R.track == track]
        ys = []
        for i, inp in enumerate(order):
            for j, fam in enumerate(["window-level", "deep"]):
                r = sub[(sub.inputs == inp) & (sub.model_family == fam)]
                if not len(r): continue
                r = r.iloc[0]; y = i + (0.18 if fam == "deep" else -0.18)
                ax.errorbar(r.F1, y, xerr=[[r.F1 - r.F1_ci_low], [r.F1_ci_high - r.F1]], fmt="o" if fam == "window-level" else "s",
                            color="C0" if fam == "window-level" else "C1", ms=4, capsize=2, lw=1)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace("BEST ", "best ").replace(" (any)", "") for o in order], fontsize=8)
        ax.invert_yaxis(); ax.axvline(0.5, color="grey", lw=0.8, ls=":"); ax.set_title(f"{track} (binary)", fontsize=9, loc="left"); ax.grid(axis="x", lw=0.3)
    axes[1].set_xlabel("second-level macro-F1 (best run per condition; 95% participant-bootstrap CI)", fontsize=8)
    axes[0].plot([], [], "o", color="C0", label="window-level classifiers"); axes[0].plot([], [], "s", color="C1", label="sequence models"); axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.02))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_f1_by_inputs.pdf")); print("fig2 saved")

def _cms(row):
    cms = []
    for k in range(5):
        for c in (f"t{k}_confusion_matrix", f"fold_{k}_confusion_matrix"):
            if c in row.index and isinstance(row[c], str): cms.append(np.array(json.loads(row[c]))); break
    return sum(cms)

def fig3(a):
    parts = []
    for p, deep in ((a.recovery_linear, False), (a.recovery_deep, True)):
        if not p: continue
        d = pd.read_csv(p); d = d[d.State == "finished"]; d["task"] = d["class"] if deep else d["binary_multiclass"]; parts.append(d)
    D = pd.concat(parts); D = D[(D.task == "multiclass") & D.avg_sec_test_f1.notna()]
    g = D.modality.str.contains("gemini")
    picks = [("best conventional", D[~g].loc[D[~g].avg_sec_test_f1.idxmax()]), ("best Gemini-inclusive", D[g].loc[D[g].avg_sec_test_f1.idxmax()])]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    labels = ["neutral", "error", "recovery"]
    for ax, (title, r) in zip(axes, picks):
        cm = _cms(r); n = cm / cm.sum(1, keepdims=True)
        ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{n[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if n[i, j] > 0.6 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"{title}\n{r.model_type if 'model_type' in r and isinstance(r.model_type, str) else r.get('model','')} · macro-F1 {r.avg_sec_test_f1:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_recovery_confusion.pdf")); print("fig3 saved")

def fig4(a):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), gridspec_kw={"width_ratios": [1.4, 1]})
    if a.base_csv:
        b = pd.read_csv(a.base_csv, usecols=[0, 1, 2, 3]); b.columns = ["frame", "part", "yb", "ym"]
        b["sec"] = b.groupby("part").cumcount() // 100
        S = b.groupby(["part", "sec"]).ym.last().unstack(); S = S.loc[S.apply(lambda r: (r == 0).idxmin() if (r == 0).any() else 0, axis=1).sort_values().index]
        axes[0].imshow(S.to_numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
        axes[0].set_xlabel("elapsed second", fontsize=8); axes[0].set_ylabel("participant (sorted by survey length)", fontsize=8); axes[0].set_title("label vs elapsed time (0 / I / II / III)", fontsize=8)
    if a.time_matched:
        T = pd.read_csv(a.time_matched); x = np.arange(len(T))
        axes[1].bar(x - 0.2, T.AUC_all, 0.4, label="all seconds"); axes[1].bar(x + 0.2, T.AUC_band, 0.4, label=f"time-matched band {T.band.iloc[0]} s")
        axes[1].set_xticks(x); axes[1].set_xticklabels(T.run_id, rotation=30, ha="right", fontsize=7); axes[1].set_ylim(0.4, 1.0); axes[1].set_ylabel("second-level AUC", fontsize=8); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_clock_confound.pdf")); print("fig4 saved")

def fig5(a):
    P = pd.read_csv(a.per_participant, index_col=0).drop(columns=["diff"], errors="ignore")
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, c in enumerate(P.columns):
        ax.scatter(np.random.normal(i, 0.05, len(P)), P[c], s=12, alpha=0.7); ax.hlines(P[c].median(), i - 0.25, i + 0.25, color="k")
    ax.set_xticks(range(len(P.columns))); ax.set_xticklabels(P.columns, fontsize=8); ax.set_ylabel("per-participant macro-F1", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig5_per_participant.pdf")); print("fig5 saved")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("fig", choices=["fig2", "fig3", "fig4", "fig5"])
    for k in ["results", "recovery_linear", "recovery_deep", "base_csv", "time_matched", "per_participant"]: ap.add_argument(f"--{k}")
    ap.add_argument("--out", default="figs"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}[a.fig](a)
