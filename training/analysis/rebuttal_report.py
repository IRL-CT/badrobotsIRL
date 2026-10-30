#!/usr/bin/env python3
"""Assemble the HRI'27 rebuttal evidence into one self-contained markdown report.

The report is written for whoever revises the paper: for every triage item
(T1-T14 in HRI27_rebuttal_triage.xlsx) it gives the reviewers' ask, what was
computed, the numbers with their provenance and unit, the current paper text the
numbers bear on (file:line in the submitted sources), and the recommended change.

Inputs (paths relative to training/analysis unless absolute):
  rebuttal/relative_controls.csv                hierarchical_escalation/rebuttal_controls.py
  rebuttal/v6_escalation_export.csv             pull_v6_escalation_export.py
  rebuttal/headline_predictions_rebuttal_{A,B}.csv   run_headline.py --only 10..16
  rebuttal/confusion_stage_absolute*.csv        confusion_second_level.py / W&B pull
  headline_predictions.csv                      published headline registry (configs 1-9)
  decay_recov_{engineered,embed}.csv            recovery_decay.py on configs 4/5 (published)
  time_matched_*.csv                            time_matched_eval.py (published analysis)
  --xlsx  HRI27_rebuttal_triage.xlsx            the triage sheet (asks, proposals, notes)

    python3 rebuttal_report.py                       # repo version: asks paraphrased
    python3 rebuttal_report.py --verbatim --out ~/2024_badirl/hri27/REBUTTAL_REPORT_FULL.md
The --verbatim version also includes the reviewers' quotes from the sheet; keep it
out of the public repository.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.join(HERE, "rebuttal")
PY = sys.executable
sys.path.insert(0, HERE)
from registry_to_seconds import load_run  # noqa: E402

STAGE_NAMES = ["survey", "Error I", "Error II", "Error III"]
CONTROL_LABEL = {
    "none": "published pipeline (reference)",
    "noise": "Gaussian noise, same session structure (paper's control, this one config)",
    "onset5": "random onset: 0–5 s prefix dropped per session",
    "onset10": "random onset: 0–10 s prefix dropped per session",
    "label_shift": "labels cyclically shifted ≥ 5 s against features, within session (R2)",
    "phase_shuffle": "seconds shuffled within each phase before windowing (R1)",
}
ONSET_LABEL = {"none": "known onset", "onset5": "onset unknown ≤ 5 s", "onset10": "onset unknown ≤ 10 s"}
BLOCK_LABEL = {"current": "current only (absolute state)", "dprev_drift": "Δprev + drift",
               "dprev_drift_dbase": "Δprev + drift + Δbase (published)"}
PAPER = "HRI27_BadRobotsIRL__was_CoRL26_.zip"   # submitted sources; paths below are inside it


# ----------------------------------------------------------------------------- helpers
def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def mean_sd(g, col="sec_f1"):
    if not len(g):
        return ""
    sd = g[col].std(ddof=1) if len(g) > 1 else 0.0
    return f"{g[col].mean():.3f} ± {sd:.3f}"


def matrix_md(C, names, counts=True):
    C = np.asarray(C, dtype=float)
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    hdr = "| true ↓ / predicted → | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
    rows = []
    for i, n in enumerate(names):
        cells = [f"{Cn[i, j]:.3f}" + (f" ({int(C[i, j])})" if counts else "") for j in range(len(names))]
        rows.append(f"| **{n}** | " + " | ".join(cells) + " |")
    return hdr + "\n" + "\n".join(rows)


def quote(text):
    return "\n".join("> " + l for l in text.strip().splitlines())


def sel(ctl, config, control, blocks="as_published", baseline_mode="windows", baseline_sec=None):
    d = ctl[(ctl.config == config) & (ctl.control == control) & (ctl.baseline_mode == baseline_mode)]
    if blocks == "dprev_drift_dbase":
        d = d[d.blocks.isin(["dprev_drift_dbase", "as_published"])]
    else:
        d = d[d.blocks == blocks]
    if baseline_sec is not None:
        d = d[d.baseline_sec == baseline_sec]
    return d.drop_duplicates(["seed"]).sort_values("seed")


def sec_level(reg, rid):
    """Second-level pooled macro-F1, fold-mean macro-F1, AUC, n seconds for a registry run."""
    df = load_run(reg, rid)
    # same convention as confusion_second_level.py / recovery_decay.py: p19nodbot sits in two test folds
    df = df.sort_values(["session", "fold", "second"]).drop_duplicates(["session", "second"], keep="first")
    yt, yp = df.y_true.astype(int), df.y_pred.astype(int)
    pooled = f1_score(yt, yp, average="macro")
    per_fold = pd.Series({k: f1_score(g.y_true.astype(int), g.y_pred.astype(int), average="macro")
                          for k, g in df.groupby("fold")})
    pc = [c for c in df.columns if c[0] == "p" and c[1:].isdigit()]
    auc = np.nan
    try:
        if len(pc) == 2:
            auc = roc_auc_score(yt, df[pc[1]])
        elif len(pc) > 2:
            auc = roc_auc_score(yt, df[pc].to_numpy(), multi_class="ovr", average="macro")
    except ValueError:
        pass
    return dict(pooled=pooled, fold_mean=per_fold.mean(), fold_sd=per_fold.std(ddof=1), auc=auc, n=len(df))


def find_reg(regs, rid):
    for reg in regs:
        if reg and os.path.exists(reg) and rid in set(pd.read_csv(reg, usecols=["run_id"]).run_id):
            return reg
    return None


def censored_median(csv):
    d = pd.read_csv(csv)
    t = d.seconds_to_flip.to_numpy(dtype=float)
    L = d.recovery_len_s.to_numpy(dtype=float)
    cens = np.isnan(t)
    t_all = np.where(cens, L, t)
    order = np.argsort(t_all, kind="stable")
    S, km, at_risk = 1.0, np.nan, len(t_all)
    for i in order:
        if not cens[i]:
            S *= 1 - 1 / at_risk
        at_risk -= 1
        if S <= 0.5 and np.isnan(km):
            km = t_all[i]
    return dict(crossers=int((~cens).sum()), n=len(d), median=np.nanmedian(t),
                iqr=(np.nanpercentile(t, 25), np.nanpercentile(t, 75)), km=km,
                still_error=d.frac_called_error.mean())


def load_triage(path):
    if not path or not os.path.exists(path):
        return {}
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Triage"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) for h in rows[0]]
    out = {}
    for r in rows[1:]:
        if r[0]:
            out[str(r[0])] = dict(zip(hdr, [("" if v is None else str(v)) for v in r]))
    return out


# ----------------------------------------------------------------------------- sections
def item_header(tid, tri, verbatim):
    t = tri.get(tid, {})
    out = [f"## {tid} — {t.get('Theme', '')}", ""]
    if t:
        out += [f"**Reviewers:** {t.get('Reviewers', '')}. **Effort (sheet):** {t.get('Effort', '')}.", "",
                f"**What is asked:** {t.get('What is asked', '')}", ""]
        if verbatim and t.get("Verbatim quotes (reviewer tagged)"):
            out += ["**Reviewer text (verbatim, from the triage sheet):**", "",
                    quote(t["Verbatim quotes (reviewer tagged)"].replace(" / ", "\n")), ""]
        if t.get("teresa comments"):
            out += [f"**Author note (sheet):** {t['teresa comments']}", ""]
    return out


def sec_overview(ctl, regs, checks, f1_exact="0.733"):
    out = ["# HRI'27 rebuttal — evidence report", "",
           "Paper: *When at First You Don't Succeed: Detecting Error, Stage, and Recovery in Successive Robot "
           f"Failure* (HRI 2027 submission; sources in `{PAPER}`, line numbers below refer to those files). "
           "Triage: `HRI27_rebuttal_triage.xlsx`, items T1–T14. Repo: `badrobotsIRL`, branch "
           "`gemini-v4-v6-sweeps`. Every number in this document is produced by a script in "
           "`training/` and written to `training/analysis/rebuttal/`; `rebuttal/README.md` maps scripts to "
           "SLURM jobs and outputs. Generated by `training/analysis/rebuttal_report.py`.", "",
           "**Units.** The paper's primary metric is second-level macro-F1 (model rows within one second "
           "reduced by majority vote), mean over five participant-disjoint folds. The relative-feature arm "
           "scores one unit per *window* (5 s, stride 2 s for the published best runs; ~330 windows over 24 "
           "participants), and the paper reports its pooled value; this report does the same. Where a number "
           "is pooled over folds rather than fold-averaged it is marked *pooled*.", "",
           "## Reproduction checks (published value vs. this re-run)", "", md_table(pd.DataFrame(checks)), "",
           "The two relative-feature runs reproduce bit-for-bit at seed 0 (the sweep used `random_state=42`; "
           "seeds 1–2 vary the forest). The two window-level runs reproduce within 0.010–0.015 F1: the sweep "
           "logged the fold-mean of second-level F1, while the registry-based analyses pool seconds over folds "
           "and de-duplicate p19nodbot, who sits in two test folds (every registry-based number in this report uses that "
           "de-duplicated set, so the SMOTE, confusion and delay numbers are on identical rows).", "",
           "## Executive summary — effect on the paper's claims", "",
           "1. **Recovery (§5.2, Tables 3, 5, 6): unchanged and strengthened.** The exact best binary recovery "
           f"model reproduces (F1 {f1_exact} vs 0.735) and yields the same delay picture as the two weaker retrained "
           "models: 22/24 participants cross at k = 3, median delay 5 s, 47 % of recovery seconds still "
           "classified as error. Treating never-crossers as censored moves medians by ≤ 1 s. SMOTE off: "
           "ΔF1 −0.007.", "",
           "2. **Detection representation (§5.1, Table 2): unchanged.** No new run touches Table 2; SMOTE "
           "on/off deltas for the headline configs are within ±0.02 (−0.03 for stage), inside fold SD.", "",
           "3. **Relative features (§5.3, Table 4, Table 5, abstract, §1, §6, §7): the stage half holds and "
           "is now better supported; the detection half (0.959) does not hold as a behavioral finding and "
           "should be re-scoped or removed.** On a fully matched pipeline the absolute-state stage model "
           "reaches 0.458 and the relative one 0.618–0.620 (+0.16; the paper's 'half smoothing, half framing' "
           "reading is wrong — it is all framing, since 0.458 ≈ Table 2's 0.469). The gain is behavioral "
           "(labels shifted within session → 0.25, chance; noise → 0.34), survives within-phase shuffling and "
           "a 5 s unknown onset, and persists over the absolute model at every onset uncertainty (+0.21, "
           "+0.21, +0.11 for known / ≤5 s / ≤10 s). It comes entirely from Δprev + drift; Δbase adds nothing "
           "for stage. For detection, current-only gives 0.676, Δprev + drift 0.673 and adding Δbase 0.959: "
           "the whole gain is the own-baseline term anchored to a session start the protocol guarantees is "
           "pre-failure, and the advantage over the absolute model shrinks from +0.28 to +0.10 (≤5 s) to "
           "+0.04 (≤10 s).", "",
           "4. **Two factual corrections to Table 5** (found while reproducing, independent of any new run): "
           "the best relative detection run is acoustic-only (25 eGeMAPS features; its W&B modality is "
           "`audio`), not 'A+G768'; and the best stage run's baseline is the first 5 s window, not '2 s', "
           "because `escalation_linear.py` converts `baseline_sec` to windows (`baseline_sec // stride_sec`, "
           "so b = 2 and b = 3 both mean one 5 s window at stride 2).", "",
           "5. **One definitional note for Table 4.** Its noise column is the best of ~1,000 noise runs in the "
           "sweep (0.703 / 0.343), i.e., a best-of-search number like the real column. A single matched "
           "noise run for the published configs scores 0.456 (detection) and 0.337 (stage); the paper's Δ is "
           "therefore conservative, not inflated.", ""]
    return out


def sec_T1(ctl, exp, tri, verbatim):
    out = item_header("T1", tri, verbatim)
    e = exp[(exp.state == "finished") & exp.avg_sec_test_f1.notna()].copy()
    for c in ["use_current", "use_delta_base", "use_delta_prev", "use_drift", "use_scalars"]:
        e[c] = e[c].fillna(False).astype(bool)
    e["noise_input"] = e["noise_input"].fillna(False).astype(bool)
    defs = {"current only (absolute state)": lambda d: d.use_current & ~d.use_delta_base & ~d.use_delta_prev & ~d.use_drift & ~d.use_scalars,
            "Δprev + drift": lambda d: ~d.use_current & ~d.use_delta_base & d.use_delta_prev & d.use_drift & ~d.use_scalars,
            "Δprev + drift + Δbase": lambda d: ~d.use_current & d.use_delta_base & d.use_delta_prev & d.use_drift & ~d.use_scalars}
    rows = []
    for name, f in defs.items():
        r = {"feature blocks": name}
        for task, lab in [("binary", "detection"), ("multiclass", "stage")]:
            for noise, nl in [(False, "real"), (True, "noise")]:
                d = e[(e.binary_multiclass == task) & (e.noise_input == noise)]; d = d[f(d)]
                r[f"{lab} {nl} best (n runs)"] = f"{d.avg_sec_test_f1.max():.3f} ({len(d)})"
            dr = e[(e.binary_multiclass == task) & (~e.noise_input)]; dr = dr[f(dr)]
            r[f"{lab} real median"] = dr.avg_sec_test_f1.median()
        rows.append(r)
    out += ["### Evidence A — the 'current only' row exists in the sweep (as swept, best over the random search)",
            "", "Exact block sets, `step` excluded, finished runs, from the uncapped W&B export "
            "(`pull_v6_escalation_export.py`; early runs with `noise_input` unset are real runs and are included).", "",
            md_table(pd.DataFrame(rows)), "",
            "This reproduces the row that was commented out of main-text Table 4 / supp Table (0.821 / 0.492 "
            "with noise 0.545 / 0.301 in the submission; noise maxima moved slightly because the sweep kept "
            "running after the export used for the paper). The middle row's paper values (0.946 / 0.614, "
            "n = 731/733) used a looser block definition; with exact block sets it is 0.946 / 0.599.", ""]
    rows = []
    for cfg, lab in [("stage_best", "Stage · RF · curated(31)+G256[vid+aud] · w 5 s · stride 2 s"),
                     ("detect_best", "Detection · logreg · acoustic(25) · w 5 s · stride 2 s")]:
        for b in ["current", "dprev_drift", "dprev_drift_dbase"]:
            g = sel(ctl, cfg, "none", b)
            rows.append({"config": lab, "feature blocks": BLOCK_LABEL[b], "F1 (mean ± SD, 3 seeds)": mean_sd(g),
                         "per-seed": ", ".join(f"{v:.3f}" for v in g.sec_f1), "AUC": g.sec_auc.mean(),
                         "n features": int(g.n_features.iloc[0])})
    out += ["### Evidence B — matched triple (R3's ask: same modality, window, model family, folds; only the "
            "feature blocks change)", "", "`rebuttal_controls.py`, item T1. Pooled window-level macro-F1.", "",
            md_table(pd.DataFrame(rows)), "",
            "Reading. **Stage:** absolute state on the 1–5 s pipeline scores 0.458, essentially Table 2's "
            "0.469, so the coarser window explains none of the gap; the change blocks (Δprev + drift) give "
            "+0.16 and Δbase adds nothing (0.620 → 0.618). **Detection:** the change blocks give nothing "
            "(0.676 → 0.673) and Δbase gives +0.29 (→ 0.959). See T2 for what that means.", "",
            "### Current paper text this bears on", "",
            "* `sections/05_results.tex` L184 (Table 4 caption): \"The first row repeats the best absolute-state "
            "result of Table 2 for reference; those models use 0.01–0.3 s windows against 1–5 s here, so the "
            "comparison is indicative.\" and L196 the row `Absolute (best of Table 2) & 0.789 & --- & --- & 0.469`.",
            "* `sections/05_results.tex` L211: \"...raises stage identification to F1 = 0.631, against 0.469 "
            "for the best absolute-state model in Table 2, and detection to 0.959, against 0.789. Part of the "
            "difference is the coarser window ... the within-table comparison is not ...\"",
            "* `supp_material.tex` L600–640: the full relative table with the 'current only' row is commented out.",
            "* Table 5 (`05_results.tex` L241–248) rows 'Stage (relative)' and 'Detection (relative)'.", "",
            "### Recommended change", "",
            "* Replace Table 4's first row with the same-pipeline absolute row (as swept: 0.821 / 0.492, noise "
            "0.553 / 0.301) and add the matched triple (Evidence B) as either a second block or a supp table; "
            "cite the sweep-best row for comparability with the sweep and the matched triple for the controlled "
            "claim.", "* Rewrite the §5.3 lead paragraph around the matched comparison: stage 0.458 → 0.620 "
            "(+0.16, all framing, from Δprev + drift); drop the sentence attributing part of the gain to the "
            "coarser window.", "* State explicitly that for detection the relative gain is the own-baseline "
            "term only (see T2 for scoping).", ""]
    return out


def sec_T2(ctl, tri, verbatim):
    out = item_header("T2", tri, verbatim)
    out += ["### Evidence A — what already existed", "",
            "* *Noise control* (paper Table 4): features replaced by Gaussian noise with sessions, windows, labels "
            "and folds unchanged; best noise run 0.703 (detection) / 0.343 (stage) vs real 0.959 / 0.631.",
            "* *Elapsed-time-only baseline* (`elapsed_time_baseline.py`, not in the paper): logistic regression on "
            "seconds-since-start alone reaches F1 0.976 / AUC 0.999 for detection and 0.876 / 0.959 for "
            "error-vs-recovery, i.e., the clock beats every model. This is a property of any fixed-script "
            "study and should be stated plainly.",
            "* *Time-matched evaluation* (`time_matched_eval.py`, not in the paper): scoring only in seconds where "
            "both classes occur across participants. Detection collapses (only 5 such seconds; AUC 0.57–0.71, "
            "CIs span 0.5); recovery survives (19 mixed seconds; AUC 0.69 engineered, 0.77 with embedding; "
            "AUC conditioned on the exact second 0.70 / 0.78). Files: "
            "`training/analysis/time_matched_v4_detection.csv`, `time_matched_v5_recovery_binary.csv`.", ""]
    for cfg, task in [("stage_best", "Stage (4-class) — RF, curated+G256, w = 5 s, stride 2 s, published F1 0.631"),
                      ("detect_best", "Detection (binary) — logistic regression, acoustic only, w = 5 s, stride 2 s, published F1 0.959")]:
        rows = []
        ref = sel(ctl, cfg, "none").sec_f1.mean()
        for c in ["none", "noise", "onset5", "onset10", "label_shift", "phase_shuffle"]:
            g = sel(ctl, cfg, c)
            rows.append({"control": CONTROL_LABEL[c], "F1 (mean ± SD, 3 seeds)": mean_sd(g), "AUC": g.sec_auc.mean(),
                         "Δ vs reference": g.sec_f1.mean() - ref,
                         "per-seed F1": ", ".join(f"{v:.3f}" for v in g.sec_f1)})
        out += [f"### Evidence B — new controls, {task}", "", md_table(pd.DataFrame(rows)), ""]
    oi = sel(ctl, "stage_best", "onset10").control_info.dropna()
    if len(oi):
        cuts = [v.get("cut_s", 0) for info in oi for fold in json.loads(info).values() for v in fold.values()]
        out += [f"Random-onset draws (10 s): mean prefix removed {np.mean(cuts):.1f} s, range {min(cuts)}–{max(cuts)} s "
                f"over {len(cuts)} participant-fold draws; the pre-failure (survey) block is 10.4 s at the median, "
                "so a 10 s cut removes most of it for many participants and a 5 s cut leaves it observable for all.", ""]
    rows = []
    for cfg, lab in [("stage_best", "stage"), ("detect_best", "detection")]:
        for b in ["current", "dprev_drift", "dprev_drift_dbase"]:
            r = {"task": lab, "feature blocks": BLOCK_LABEL[b]}
            for c in ["none", "onset5", "onset10"]:
                r[ONSET_LABEL[c]] = mean_sd(sel(ctl, cfg, c, b))
            rows.append(r)
    out += ["### Evidence C — absolute vs relative blocks on the SAME shifted sessions (3 seeds)", "",
            "Separates 'the test set got harder' from 'the relative features lost their anchor'. "
            "`rebuttal_controls.py`, items T1 + T2b.", "", md_table(pd.DataFrame(rows)), "",
            "### Reading", "",
            "* **The stage signal is behavioral.** With real features but labels cyclically shifted within each "
            "session (phase lengths and order preserved), F1 falls to 0.25 — chance for four classes — below "
            "even the Gaussian-noise control (0.34, which still sees the structural cue that early windows have "
            "small deltas). A model exploiting phase base rates or elapsed time would score well above chance "
            "here; it does not.",
            "* **Within-phase order does not matter** (phase shuffle −0.03 for stage): the relative features read "
            "a phase-level behavioral state, not a fine-grained within-phase trajectory. R1's prediction was "
            "that the drift features would lose efficacy; they do not, which argues against a pseudo-clock "
            "reading but also means the features are not capturing escalation *within* an error instance.",
            "* **Stage survives an uncertain onset and keeps its margin over absolute state.** Known onset "
            "0.618 vs 0.458; onset unknown within 5 s 0.597 vs 0.391; within 10 s 0.457 vs 0.349. The 10 s "
            "drop is shared by the absolute model (the survey class largely disappears), so it reflects the "
            "test set, not the relative framing.",
            "* **Detection is a different story.** The published 0.959 is entirely the Δbase term (Evidence B of "
            "T1) and depends on the first seconds being pre-failure: with the onset unknown within 5 s it falls "
            "to 0.717 (absolute 0.619), within 10 s to 0.575 (absolute 0.535), close to the label-shift floor "
            "(0.567). The within-phase shuffle also costs it 0.20. The honest description is 'given a baseline "
            "window known to precede the first failure, deviation from it flags the error at F1 ≈ 0.96'.", "",
            "### Current paper text this bears on", "",
            "* Abstract (`00_hri_main.tex` L181) and §1 (`01_introduction.tex` L19) pair stage with 'change "
            "relative to the user's own earlier behavior' — stage only, fine. §5.3 (`05_results.tex` L211) adds "
            "'and detection to 0.959, against 0.789' and 'Both exceed the noise control by 0.25 on detection and "
            "0.27–0.29 on stage, so the gain is not an artifact of the session's fixed structure.'",
            "* §6 (`06_discussion.tex` L12–14) 'What generalizes is a frame of reference' — stage only; keep, "
            "add the scoping.", "* §6 Limitations (`06_discussion.tex` L18) 'Detection validation: ... the design "
            "cannot fully separate behavior that changes because of the failures from behavior that changes as "
            "a session goes on.'", "* §3.6 Noise control paragraph (`03_methods.tex` L638).", "",
            "### Recommended change", "",
            "* Add the three controls to §3.6 (one sentence each) and a compact table (Evidence B for stage, "
            "Evidence C) to §5.3 or the supplement.", "* Keep the stage claim, scoped: 'behavior carries stage "
            "information beyond session position; the model needs the session start to within a few seconds, "
            "and the advantage over absolute state holds at every onset uncertainty we tested'.",
            "* Re-scope or remove the detection 0.959: report it as a known-onset calibration result (own-baseline "
            "subtraction when the baseline precedes the first failure), not as evidence that relative features "
            "help detection.", "* State the elapsed-time result in Limitations as a property of scripted-failure "
            "studies, and cite the time-matched evaluation as the reason recovery is the stronger contribution.", ""]
    return out


def sec_T3(regs, tri, verbatim):
    out = item_header("T3", tri, verbatim)
    out += ["### Facts about the pipeline (verified in code)", "",
            "* Order of operations in every window-level script (`temp_allscripts/*_model.py`, `recovery/*_model.py`): "
            "load 100 fps frames → sliding-window aggregation (`apply_windowing`) → participant-level split "
            "(`create_data_splits`) → SMOTE on the **training fold's window-level rows only** → fit. Validation "
            "and test rows are never resampled. SMOTE therefore interpolates between aggregated window vectors "
            "of training participants, not between raw adjacent frames; `k_neighbors = min(5, n_minority − 1)`, "
            "`random_state = 42`.", "* SMOTE is used only by the window-level classifiers (RF, SGD, SVM, MLP, "
            "k-NN). The sequence models (GRU, LSTM, Transformer) use no resampling. The relative-feature arm "
            "(`escalation_linear.py`, Table 4 / Table 5 stage and relative rows) uses **no SMOTE**: the random "
            "forest uses `class_weight='balanced'`, logistic regression / SGD / MLP are unweighted.", ""]
    pairs = [("detection, best embedding-inclusive modality (config 2, Table 2 best row)", "headline_2_detect_bin_embed_incl", "headline_13_detect_bin_embed_incl_nosmote"),
             ("detection, best engineered (config 1)", "headline_1_detect_bin_engineered", "headline_14_detect_bin_engineered_nosmote"),
             ("recovery binary, best engineered (config 4, delay model)", "headline_4_recov_bin_engineered", "headline_15_recov_bin_engineered_nosmote"),
             ("recovery binary, exact best (splendid-sweep-136)", "headline_10_recov_bin_best_embed_exact", "headline_12_recov_bin_best_embed_exact_nosmote"),
             ("stage, exact best absolute (true-sweep-651)", "headline_11_stage_best_abs_exact", "headline_16_stage_best_abs_exact_nosmote")]
    rows = []
    for lab, a, b in pairs:
        r = {"model": lab}
        for tag, rid in [("SMOTE on", a), ("SMOTE off", b)]:
            reg = find_reg(regs, rid)
            m = sec_level(reg, rid) if reg else None
            r[f"{tag} F1 pooled"] = m["pooled"] if m else np.nan
            r[f"{tag} F1 fold-mean ± SD"] = f"{m['fold_mean']:.3f} ± {m['fold_sd']:.3f}" if m else ""
            r[f"{tag} AUC"] = m["auc"] if m else np.nan
        r["Δ F1 pooled (off − on)"] = r["SMOTE off F1 pooled"] - r["SMOTE on F1 pooled"]
        rows.append(r)
    out += ["### Evidence — SMOTE on vs off, identical configs otherwise (`run_headline.py` configs 1/2/4 vs 13/14/15, 10 vs 12, 11 vs 16)",
            "", md_table(pd.DataFrame(rows)), "",
            "All deltas are inside the fold-level SD (0.02–0.09). SMOTE neither creates nor removes the result.", "",
            "### Current paper text", "", "* `03_methods.tex` L627: \"Training folds are rebalanced with SMOTE; "
            "validation and test folds are never resampled.\"", "",
            "### Recommended change", "", "* Extend that sentence: SMOTE is applied after windowing, to the "
            "window-level feature vectors of the training participants only, by the window-level classifiers; "
            "sequence models and the relative-feature models use no resampling (class weights for RF).",
            "* Optionally one sentence or a supp row: switching SMOTE off changes the headline F1 by −0.03 to "
            "+0.02 (table above). The author note on the sheet suggests the clarification alone may suffice; the "
            "numbers are available either way.", ""]
    return out


def sec_T4_T5(regA, tri, verbatim):
    out = item_header("T4", tri, verbatim)
    out += ["### Facts", "", "* Author note (sheet): the recovery phase contains only seconds in which the "
            "participant was alone in the room; the participant does not turn to the door. No gaze annotation "
            "was run (author's decision).", "* Methods already say recording stopped when the researcher "
            "re-entered (`03_methods.tex` L557) and Limitations already call never-crossers censored in effect "
            "(`06_discussion.tex` L18).", ""]
    rows = []
    pub = [("F+A+P+C (config 4, F1 0.716, published Table 6)", os.path.join(HERE, "decay_recov_engineered.csv")),
           ("F+A+P+G128 (config 5, F1 0.708, published Table 6)", os.path.join(HERE, "decay_recov_embed.csv"))]
    for lab, csv in pub:
        if os.path.exists(csv):
            m = censored_median(csv)
            rows.append({"model": lab, "k": 3, "crosses": f"{m['crossers']}/{m['n']}",
                         "median delay, crossers (IQR)": f"{m['median']:.0f} s ({m['iqr'][0]:.0f}–{m['iqr'][1]:.0f})",
                         "Kaplan–Meier median (never = censored at end of recording)": f"{m['km']:.0f} s",
                         "still error": f"{m['still_error']:.0%}"})
    out += ["### Evidence — right-censored medians (never-crossers censored at the end of their recovery recording)",
            "", md_table(pd.DataFrame(rows)), "",
            "### Recommended change", "", "* One sentence in §3.5 or §5.2: the recovery phase contains only "
            "seconds in which the participant was alone; the recording ends at the researcher's return.",
            "* Report 'never' as censored (the Kaplan–Meier medians are 1 s above the crosser medians) and keep "
            "'hangover' wording out.", ""]
    # ---- T5
    out += item_header("T5", tri, verbatim)
    rid = "headline_10_recov_bin_best_embed_exact"
    rows = []
    for lab, csv in pub:
        if os.path.exists(csv):
            m = censored_median(csv)
            rows.append({"model": lab, "k": 3, "crosses": f"{m['crossers']}/{m['n']}",
                         "median delay (IQR)": f"{m['median']:.0f} s ({m['iqr'][0]:.0f}–{m['iqr'][1]:.0f})",
                         "KM median": f"{m['km']:.0f} s", "still error": f"{m['still_error']:.0%}"})
    if regA and find_reg([regA], rid):
        m = sec_level(regA, rid)
        out += ["### Evidence", "",
                f"The sweep-best binary recovery run (splendid-sweep-136: RF 200 trees depth 10, F+A+P+G128 "
                f"[vid+aud], window 10 frames stride 10, mean aggregation, window statistics on; W&B second-level F1 "
                f"0.735 / AUC 0.847) was retrained with its exact hyperparameters (`run_headline.py` config 10): "
                f"second-level F1 {m['pooled']:.3f} pooled, {m['fold_mean']:.3f} ± {m['fold_sd']:.3f} over folds, "
                f"AUC {m['auc']:.3f}. The two models used in the submission (configs 4 and 5) had grid-midpoint "
                "hyperparameters, which is why they scored 0.716 / 0.708.", ""]
        for k in (2, 3, 4):
            o = os.path.join(REB, f"decay_best_exact_k{k}.csv")
            if not os.path.exists(o):
                subprocess.run([PY, os.path.join(HERE, "recovery_decay.py"), "--registry", regA, "--run_id", rid,
                                "--k", str(k), "--out", o], check=True, capture_output=True)
            mk = censored_median(o)
            rows.append({"model": f"F+A+P+G128 exact best (F1 {m['pooled']:.3f})", "k": k,
                         "crosses": f"{mk['crossers']}/{mk['n']}",
                         "median delay (IQR)": f"{mk['median']:.0f} s ({mk['iqr'][0]:.0f}–{mk['iqr'][1]:.0f})",
                         "KM median": f"{mk['km']:.0f} s", "still error": f"{mk['still_error']:.0%}"})
        out += [md_table(pd.DataFrame(rows)), ""]
        d = pd.read_csv(os.path.join(REB, "decay_best_exact_k3.csv"))
        d["seconds_to_flip"] = d.seconds_to_flip.map(lambda v: "never" if pd.isna(v) else f"{v:.0f}")
        d["frac_called_error"] = d.frac_called_error.map(lambda v: f"{v:.0%}")
        d["mean_p_recovery"] = d.mean_p_recovery.round(2)
        d.columns = ["participant", "recovery length (s)", "delay to 3 consecutive s > 0.5", "fraction of recovery seconds called error", "mean P(recovery)"]
        out += ["Per participant, exact best model, k = 3 (`rebuttal/decay_best_exact_k3.csv`):", "", md_table(d), ""]
        tr = pd.read_csv(os.path.join(REB, "decay_best_exact_k3_trajectory.csv"))
        tr = tr[tr["count"] >= 7].copy()
        tr["t_since_ack"] = tr.t_since_ack.astype(int).astype(str); tr["count"] = tr["count"].astype(int).astype(str)
        out += ["Mean P(recovery) by second since acknowledgment (exact best model; seconds with ≥ 7 participants):", "",
                md_table(tr.rename(columns={"t_since_ack": "s since ack", "mean": "mean P(recovery)", "std": "SD", "count": "n participants"}), "{:.2f}"), "",
                "P(recovery) first exceeds 0.5 on average at 5 s after the acknowledgment, which is the median "
                "delay by the run-length rule as well.", ""]
    out += ["### Current paper text", "", "* `05_results.tex` L151–167: Table 6 (`tab:decay`) and the paragraph "
            "'The delay is measurable.' ending '...so the delays are indicative rather than a property of the best "
            "model'.", "* `supp_material.tex` 'Recovery delay: run-length check and runs used' (Table `tab:decay_k`).", "",
            "### Recommended change", "", "* Add the exact best model as a row of Table 6 (or replace the F+A+P+G128 "
            "row with it) and delete the 'indicative rather than a property of the best model' sentence.",
            "* Move the k = 2/3/4 sensitivity into the main text as one sentence: crossers vary with k "
            "(23/22/17 of 24); the median delay (5 s) and the fraction still read as error (47 %) do not.", ""]
    return out


def sec_text_only(tri, verbatim):
    out = []
    facts = {
        "T6": ["* The only independent state evidence is Liu et al.'s annotation (HRI'25 LBR): frustration displays "
               "escalate across errors; the paper's §3 says 'nine of their 26 participants stopped re-engaging "
               "altogether after Error II or III' while the triage sheet says '7 of 26 abandon after Error III' — "
               "**reconcile against Liu et al. before quoting either.**",
               "* Nothing computed here establishes a participant's internal state; the confusion asymmetry (recovery "
               "→ error 32–47 %, recovery → survey 4–7 %) is a statement about what a cross-user model reads.",
               "",
               "Recommended change: rephrase abstract L181 ('the behavioral signature of failure persists past the "
               "robot repair'), §5.2 L147 ('the user's behavior does not return to how it looked before the first "
               "failure; what it still resembles is the failure itself'), §6 L5–6 ('Repair does not reset the user'; "
               "'will ... interact with a user who still looks like someone being misunderstood') and §7 to 'is read "
               "as / resembles ongoing failure to a cross-user model'; one sentence tying to Liu et al.'s annotated "
               "frustration and abandonment."],
        "T7": ["* Supporting numbers from T10: the relative model recognises Error I at 0.60 recall and Error III at "
               "0.65, so user-perceived stage is observable from the user; the weak boundary is II vs III.",
               "* Liu et al. per-stage behavior (already cited in §3 L561): reformulation after I, slower/simpler "
               "prompts and demanding tone after II, abandonment after III; repair-strategy literature cited "
               "(Esterwood, Honig).", "",
               "Recommended argument (sheet): (1) the robot cannot count — it does not know which of its utterances "
               "the user experienced as a failure, so user-perceived stage is only observable from the user; (2) "
               "stage → policy: I = re-prompt/clarify, II = change strategy or modality, III = hand off; (3) stage is "
               "the quantitative form of the escalation Liu et al. described."],
        "T8": ["* Known: HelperBot is a Nodbot (2-axis servo motion, speaker; `03_methods.tex` L557 footnote); camera "
               "below the robot at robot POV; Liu et al. report participants looking at and gesturing toward the "
               "robot.", "* **Needed from the authors:** what the robot physically did at each utterance (nod, turn, "
               "nothing).", "* Numbers available: single-modality detection F 0.735, P 0.724, A 0.723 (Table 2); "
               "the best relative detection model is acoustic-only (T1/T2), which is exactly what would transfer to "
               "a voice assistant; gaze-to-robot and body orientation (F, P) have no voice-assistant analogue.", "",
               "Recommended change: describe the embodied behavior; concede the acoustic-only results as the part "
               "that transfers to voice assistants, as a scoped contribution."],
        "T9": ["* No computation. Position the contribution as new knowledge: (i) first cross-user evaluation of "
               "successive failure and first treatment of the post-repair phase as a recognition target; (ii) a "
               "controlled representation finding — relative-to-self transfers for stage, absolute state does not "
               "(now with the matched triple and the label-shift control from T1/T2, which person-normalization work "
               "could not run because it lacks an unlabeled protocol-supplied baseline); (iii) the deployment-relevant "
               "parity result for detection. Position against ERR@HRI (single error) and REPAIR-Bench (independent "
               "trials, predicts strategy not user recovery). Related work §2 L15–16 already frames the "
               "relative-to-self idea."],
        "T12": ["* No new run. The sequence-model null result is confounded with the 3 s context cap; a 10 s context "
                "would let sequence models read session position and reopen T2 for them. Recommended: soften "
                "`05_results.tex` L260 and `06_discussion.tex` L16 to 'within a 3 s context, sequence models do not "
                "outperform window-level models' and drop 'inverts the within-user picture' as a finding."],
        "T13": ["* No computation. One paragraph: why this is the only dataset with successive dependent failures "
                "plus a post-repair phase; what is expected to transfer (the recovery pattern, the relative-feature "
                "principle for stage) vs not (absolute numbers); tone down 'homes and hospitals' in §1 L19 if "
                "needed. The sheet notes space is tight."],
        "T14": ["* No computation. Add 'We did not collect new data; we re-analyze the recordings of Liu et al.' to "
                "the abstract and the first line of §3 (`03_methods.tex` L557 currently reads 'We use the dataset "
                "of Liu et al.')."],
    }
    for tid in ["T6", "T7", "T8", "T9", "T12", "T13", "T14"]:
        out += item_header(tid, tri, verbatim)
        out += ["### Facts and recommendation", ""] + facts[tid] + [""]
        prop = tri.get(tid, {}).get("Proposed response", "")
        if prop:
            out += [f"**Proposed response (sheet):** {prop}", ""]
    return out


def sec_T10(ctl, tri, verbatim):
    out = item_header("T10", tri, verbatim)
    out += ["**Verification of the author note:** the relative-feature sweep never logged confusion matrices "
            "(checked across all 12,236 escalation-arm runs on W&B); the absolute arm logs per-fold matrices at "
            "window-row level (`t{fold}_confusion_matrix`). The relative matrix below comes from the exact re-run; "
            "the absolute one from re-running true-sweep-651 through `run_headline.py` and pooling at second "
            "level (`confusion_second_level.py`); the W&B row-level pooled matrix is given as a cross-check.", ""]
    ref = sel(ctl, "stage_best", "none")
    C0 = np.array(json.loads(ref.iloc[0].confusion))
    out += [f"### Relative features — best stage run (serene-sweep-42 reproduced, seed 0, F1 {ref.iloc[0].sec_f1:.3f}; "
            f"{int(ref.iloc[0].n_eval_units)} windows of 5 s at stride 2 s, 24 participants)", "",
            matrix_md(C0, STAGE_NAMES), ""]
    Cm = np.mean([(lambda c: c / np.maximum(c.sum(1, keepdims=True), 1))(np.array(json.loads(c), dtype=float))
                  for c in ref.confusion], axis=0)
    out += ["Mean row-normalised over the 3 seeds:", "", matrix_md(Cm, STAGE_NAMES, counts=False), ""]
    fa = os.path.join(REB, "confusion_stage_absolute.csv")
    if os.path.exists(fa):
        a = pd.read_csv(fa).iloc[0]
        out += [f"### Absolute state — best stage run (true-sweep-651 reproduced; second level, {int(a.n_seconds)} seconds, "
                f"24 participants, macro-F1 {a.macro_F1:.3f} pooled)", "",
                matrix_md(np.array(json.loads(a.confusion_counts)), STAGE_NAMES), ""]
    fw = os.path.join(REB, "confusion_stage_absolute_wandb_rowlevel.csv")
    if os.path.exists(fw):
        w = pd.read_csv(fw)
        for _, r in w.iterrows():
            out += [f"Cross-check, W&B per-fold matrices pooled, window-row level, {r.run} ({int(r.n_rows)} rows):", "",
                    matrix_md(np.array(json.loads(r.confusion_counts)), STAGE_NAMES, counts=False), ""]
    out += ["### Reading", "",
            "* Relative framing lifts every diagonal: survey 0.69 → 0.83, Error I 0.41 → 0.60, Error II 0.37 → 0.43, "
            "Error III 0.37 → 0.65.", "* It removes the non-adjacent confusions: Error III → Error I falls from "
            "0.28 to 0.08, Error I → survey from 0.18 to 0.04, Error II → survey from 0.20 to 0.00.",
            "* The boundary that remains weak is **Error II vs III**: 0.42 of Error II windows are read as Error III "
            "(and 0.22 of Error III as II). Error I is resolved from both neighbours. So the answer to R1 is: "
            "relative features resolve Error I and separate all error stages from the survey; they do not resolve "
            "the II/III boundary. Note this is the opposite pattern to Liu et al.'s action-unit result (II ≠ III but "
            "I = II), which the §5.1 text currently invokes to explain the absolute model's confusions.", "",
            "### Current paper text", "", "* `05_results.tex` L68: '...consistent with the finding of Liu et al. "
            "that frustration-linked action units separate Error II from III but not I from II.'", "",
            "### Recommended change", "", "* Add both matrices to the supplement (or the relative one to the main "
            "text next to Table 4) and say which boundary moved (I resolved; II/III not).", ""]
    return out


def sec_T11(ctl, exp, tri, verbatim):
    out = item_header("T11", tri, verbatim)
    out += ["### Correction found in the pipeline", "",
            "`escalation_linear.py` L170: `baseline_win = max(1, cfg.baseline_sec // max(1, cfg.stride_sec))`, and the "
            "baseline is the mean of the first `baseline_win` *windows* (L109). For the published best stage run "
            "(w = 5 s, stride 2 s): b = 2 and b = 3 → 1 window = seconds 0–4; b = 5 → 2 windows = seconds 0–6. For "
            "the best detection run (same geometry, b = 5) the baseline is seconds 0–6. Only at w = 1 does the swept "
            "value equal a calibration time. **Table 5 ('baseline 2 s' / 'baseline 5 s') and §3.3 / supp 'Relative "
            "features' (b ∈ {2,3,5} s) must be corrected**, and R1's cold-start question has to be answered with the "
            "seconds-based sweep below, not with the swept parameter.", ""]
    e = exp[(exp.state == "finished") & exp.avg_sec_test_f1.notna()].copy()
    e["noise_input"] = e["noise_input"].fillna(False).astype(bool)
    for c in ["use_delta_base", "use_scalars"]:
        e[c] = e[c].fillna(False).astype(bool)
    d = e[(~e.noise_input) & e.use_delta_base & ~e.use_scalars]
    rows = []
    for b in sorted(d.baseline_sec.dropna().unique()):
        r = {"baseline_sec (as swept; windows = b // stride)": str(int(b))}
        for task, lab in [("binary", "detection"), ("multiclass", "stage")]:
            g = d[(d.binary_multiclass == task) & (d.baseline_sec == b)]
            r[f"{lab} best"] = g.avg_sec_test_f1.max(); r[f"{lab} median"] = g.avg_sec_test_f1.median(); r[f"{lab} n runs"] = str(len(g))
        rows.append(r)
    out += ["### Evidence A — as swept (all finished real runs with Δbase on, any window)", "", md_table(pd.DataFrame(rows)), "",
            "Flat within search noise; uninterpretable as calibration time for the reason above.", ""]
    rows = []
    for cfg, lab in [("stage_best", "stage"), ("detect_best", "detection")]:
        for b in (1, 2, 3, 5):
            g = sel(ctl, cfg, "none", "as_published", "seconds", b)
            rows.append({"task": lab, "baseline = first b observed seconds": str(b), "F1 (mean ± SD, 3 seeds)": mean_sd(g),
                         "per-seed": ", ".join(f"{v:.3f}" for v in g.sec_f1), "AUC": g.sec_auc.mean()})
        g = sel(ctl, cfg, "none")
        rows.append({"task": lab, "baseline = first b observed seconds": "published (first window, seconds 0–4" +
                     (" / 0–6)" if cfg == "detect_best" else ")"), "F1 (mean ± SD, 3 seeds)": mean_sd(g),
                     "per-seed": ", ".join(f"{v:.3f}" for v in g.sec_f1), "AUC": g.sec_auc.mean()})
    out += ["### Evidence B — seconds-based baseline on the published best configs (`rebuttal_controls.py`, item T11)",
            "", md_table(pd.DataFrame(rows)), "",
            "### Reading", "", "* **Stage is insensitive to the calibration length**: 1 s already gives 0.645 "
            "(the best of the four), consistent with T1's finding that Δbase contributes nothing for stage. The "
            "minimum cold-start time for stage is one second of observation, and in practice the Δprev / drift "
            "blocks need one previous window (5 s) before they are informative.",
            "* **Detection wants the longest baseline**: 0.91–0.92 at 1–3 s, 0.967 at 5 s. A 1 s baseline is within "
            "0.05 of the best; but see T2 for why this number is a known-onset result.", "",
            "### Current paper text", "", "* Table 5 (`05_results.tex` L242, L247): 'baseline 2 s' and 'baseline 5 s'.",
            "* §3.3 (`03_methods.tex` L610) and supp 'Relative Features': 'the participant's first b ∈ {2,3,5} s'.", "",
            "### Recommended change", "", "* Fix the baseline description (first window(s), i.e., 5 s / 7 s for the "
            "published runs) in Table 5, §3.3 and the supplement.", "* Report Evidence B as the answer to R1: stage "
            "F1 0.62–0.65 for any baseline from 1 to 5 s; the shortest b within 0.02 of the best is 1 s.", ""]
    return out


def sec_checklist():
    return ["## Checklist of paper edits implied by this report", "",
            "| # | location | change | item |", "|---|---|---|---|",
            "| 1 | `05_results.tex` Table 4 (L182–208) | replace 'Absolute (best of Table 2)' row with the same-pipeline absolute row(s) (as swept 0.821/0.492; matched 0.676/0.458); add the matched triple | T1 |",
            "| 2 | `05_results.tex` §5.3 L211 | rewrite: stage 0.458 → 0.620 on a matched pipeline, all framing; remove the smoothing caveat; re-scope or drop detection 0.959 | T1, T2 |",
            "| 3 | `05_results.tex` Table 5 L241–248 | 'A+G768' → acoustic only (25 features); 'baseline 2 s' → first 5 s window; 'baseline 5 s' → first two windows (7 s) | T11, T1 |",
            "| 4 | `03_methods.tex` L610 + supp 'Relative Features' | define the baseline correctly; add the three temporal controls to §3.6 (L638) | T11, T2 |",
            "| 5 | `03_methods.tex` L627 | SMOTE sentence: after windowing, training participants' window rows only, window-level classifiers only; none for sequence or relative models | T3 |",
            "| 6 | `05_results.tex` Table 6 L151–167 | add / substitute the exact best recovery model row (22/24, 5 s, 47 %); delete 'indicative rather than a property of the best model'; one sentence on k = 2/3/4 | T5 |",
            "| 7 | `03_methods.tex` L634 or §5.2 | recovery seconds are those with the participant alone; 'never' is censored | T4 |",
            "| 8 | supplement | both 4×4 confusion matrices; say Error I resolved, II/III not | T10 |",
            "| 9 | supplement or §5.3 | T2 controls table and onset ladder | T2 |",
            "| 10 | abstract L181, §5.2 L147, §6 L5–6, §7 | 'remain unrecovered' → 'is read as / resembles failure to a cross-user model' | T6 |",
            "| 11 | §6 L12–14, §7 | scope the frame-of-reference claim to stage with a known session start (to within ~5 s); say the gain is from Δprev + drift | T2 |",
            "| 12 | §5.4 L260, §6 L16 | soften the sequence-model statement to 'within a 3 s context' | T12 |",
            "| 13 | §1 / §6 | practical value of stage; embodiment description (needs author input); contribution framing; dataset scope | T7, T8, T9, T13 |",
            "| 14 | abstract, §3 L557 | 'We did not collect new data; we re-analyze the recordings of Liu et al.' | T14 |",
            "| 15 | §6 Limitations L18 | state the elapsed-time baseline (F1 0.976 detection) and the time-matched result (detection collapses, recovery survives) | T2 |", ""]


def sec_appendix(ctl):
    out = ["## Appendix — every control run (pooled window-level F1 / AUC, 95 % participant-bootstrap CI on F1)", "",
           "Source: `rebuttal/relative_controls.csv` (confusion matrices and per-participant onset cuts are in the CSV).", ""]
    d = ctl.copy()
    d["control"] = d.control.map(lambda c: c)
    d["CI"] = d.apply(lambda r: f"[{r.sec_f1_ci_low:.3f}, {r.sec_f1_ci_high:.3f}]", axis=1)
    d["fold F1 (avg)"] = d.avg_test_f1
    cols = ["item", "config", "control", "blocks", "baseline_mode", "baseline_sec", "seed", "sec_f1", "CI", "sec_auc", "fold F1 (avg)", "n_eval_units", "n_features"]
    d = d.sort_values(["config", "item", "control", "blocks", "baseline_mode", "baseline_sec", "seed"])[cols]
    out += [md_table(d), "", "## Appendix — how to regenerate", "", "```",
            "cd training/hierarchical_escalation && sbatch rebuttal_controls.sub      # 114 runs, CPU, ~15 min",
            "cd training/analysis && python3 pull_v6_escalation_export.py              # W&B export of the relative arm",
            "cd training/analysis && sbatch headline_rebuttal_A.sub && sbatch headline_rebuttal_B.sub   # configs 10-16",
            "cd training/analysis && python3 confusion_second_level.py --registry rebuttal/headline_predictions_rebuttal_A.csv \\",
            "    --run_id headline_11_stage_best_abs_exact --names survey,ErrorI,ErrorII,ErrorIII --out rebuttal/confusion_stage_absolute.csv",
            "cd training/analysis && python3 rebuttal_report.py [--verbatim --xlsx <triage.xlsx> --out <path>]",
            "```", "Interpreter: `/share/ju/conda_virtualenvs/badidea/bin/python3` (conda env `badidea`).", ""]
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REB, "REBUTTAL_RESULTS.md"))
    ap.add_argument("--xlsx", default=os.path.expanduser("~/2024_badirl/hri27/HRI27_rebuttal_triage.xlsx"))
    ap.add_argument("--verbatim", action="store_true", help="include the reviewers' verbatim quotes (not for the public repo)")
    a = ap.parse_args()

    ctl = pd.read_csv(os.path.join(REB, "relative_controls.csv"))
    exp = pd.read_csv(os.path.join(REB, "v6_escalation_export.csv"))
    regA = os.path.join(REB, "headline_predictions_rebuttal_A.csv")
    regB = os.path.join(REB, "headline_predictions_rebuttal_B.csv")
    regP = os.path.join(HERE, "headline_predictions.csv")
    tri = load_triage(a.xlsx)

    checks = []
    s0 = sel(ctl, "stage_best", "none"); d0 = sel(ctl, "detect_best", "none")
    checks.append({"run": "serene-sweep-42 (stage, relative; Table 5)", "published": "F1 0.631, AUC 0.845",
                   "re-run": f"F1 {s0.iloc[0].sec_f1:.3f}, AUC {s0.iloc[0].sec_auc:.3f} (seed 0); seeds 1–2: " + ", ".join(f"{v:.3f}" for v in s0.sec_f1.iloc[1:]), "how": "rebuttal_controls.py"})
    checks.append({"run": "fast-sweep-2151 (detection, relative; Table 5)", "published": "F1 0.959, AUC 0.981",
                   "re-run": f"F1 {d0.iloc[0].sec_f1:.3f}, AUC {d0.iloc[0].sec_auc:.3f}", "how": "rebuttal_controls.py"})
    for rid, pub, lab in [("headline_10_recov_bin_best_embed_exact", "F1 0.735 (fold-mean), AUC 0.847", "splendid-sweep-136 (recovery binary; Tables 3, 5)"),
                          ("headline_11_stage_best_abs_exact", "F1 0.469 (fold-mean), AUC 0.695", "true-sweep-651 (stage, absolute; Tables 2, 5)")]:
        reg = find_reg([regA], rid)
        if reg:
            m = sec_level(reg, rid)
            checks.append({"run": lab, "published": pub, "re-run": f"F1 {m['pooled']:.3f} pooled / {m['fold_mean']:.3f} ± {m['fold_sd']:.3f} fold-mean, AUC {m['auc']:.3f}", "how": "run_headline.py configs 10 / 11"})

    parts = []
    m10 = sec_level(regA, "headline_10_recov_bin_best_embed_exact") if find_reg([regA], "headline_10_recov_bin_best_embed_exact") else None
    parts += sec_overview(ctl, [regA, regB, regP], checks, f"{m10['pooled']:.3f}" if m10 else "n/a")
    parts += sec_T1(ctl, exp, tri, a.verbatim)
    parts += sec_T2(ctl, tri, a.verbatim)
    parts += sec_T3([regA, regB, regP], tri, a.verbatim)
    parts += sec_T4_T5(regA, tri, a.verbatim)
    parts += sec_text_only(tri, a.verbatim)   # T6-T9, T12-T14
    parts += sec_T10(ctl, tri, a.verbatim)
    parts += sec_T11(ctl, exp, tri, a.verbatim)
    parts += sec_checklist()
    parts += sec_appendix(ctl)
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print(f"-> {a.out}  ({sum(len(p) for p in parts)//1000}k chars, verbatim={a.verbatim})")


if __name__ == "__main__":
    main()
