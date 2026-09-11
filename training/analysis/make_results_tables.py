#!/usr/bin/env python3
"""Build the results-by-track table (Supp D; source for Fig. 2) from W&B exports.

Usage:
  python3 make_results_tables.py --detection_linear X.csv --detection_deep X.csv \
      --recovery_linear X.csv --recovery_deep X.csv [--hier X.csv] --out results_by_track.csv

Rules encoded here (keep in sync with the paper's §4.5):
  * metric unit is second-level (avg_sec_test_*); deep AUC uses avg_sec_test_auc if it is not
    the 0.5 placeholder, otherwise falls back to frame-level avg_test_auc and marks it 'frame'.
  * one row per (track, task, model family, input set) = the best finished run by second-level F1,
    with CI, per-fold values, n runs and median F1 in the condition.
  * hierarchical/escalation rows (if --hier given) are appended and flagged provisional.
"""
import argparse, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")

MP = {"facial": "F", "audio": "A", "pose": "P", "text": "T", "cosine": "C", "gemini": "G"}
CONV = ["facial","audio","pose","text","cosine","facial_audio","pose_facial","pose_audio","facial_text","audio_text",
        "pose_text","facial_cosine","audio_cosine","pose_cosine","pose_facial_audio","facial_audio_text","facial_audio_cosine",
        "pose_facial_text","pose_facial_cosine","pose_audio_text","pose_audio_cosine","pose_facial_audio_text","pose_facial_audio_cosine"]
GEM = ["gemini","facial_gemini","audio_gemini","pose_gemini","text_gemini","cosine_gemini","facial_audio_gemini","pose_facial_gemini",
       "pose_audio_gemini","pose_facial_audio_gemini","facial_text_gemini","audio_text_gemini","pose_text_gemini","facial_audio_text_gemini",
       "pose_facial_text_gemini","pose_audio_text_gemini","pose_facial_audio_text_gemini"]

def has_g(m): return "gemini" in str(m)
def label(m): return "+".join(MP[p] for p in str(m).split("_") if p in MP)

def folds(r, metric):
    out = []
    for k in range(5):
        for c in (f"t{k}_{metric}", f"fold_{k}_metrics.{metric}", f"fold_{k}_{metric}"):
            if c in r.index and pd.notna(r[c]): out.append(float(r[c])); break
    return np.array(out)

def prep(path, deep, track):
    if path is None: return pd.DataFrame()
    d = pd.read_csv(path); d = d[d.State == "finished"].copy()
    d["f1"] = d["avg_sec_test_f1"]
    if deep:
        d["task"] = d["class"]; d["mtype"] = d["model"]; d["family"] = "deep"
        sec_ok = d["avg_sec_test_auc"].notna() & (d["avg_sec_test_auc"] != 0.5)
        d["auc"] = np.where(sec_ok, d["avg_sec_test_auc"], d["avg_test_auc"]); d["auc_unit"] = np.where(sec_ok, "second", "frame")
        d["dims"] = d["gemini_dims"].astype("Int64").astype(str).str.replace("3072", "full")
        d["fusion"] = d.get("fusion_type", ""); d["seq"] = d.get("sequence_length", "")
    else:
        d["task"] = d["binary_multiclass"]; d["mtype"] = d["model_type"]; d["family"] = "window-level"
        d["auc"] = d["avg_sec_test_auc"]; d["auc_unit"] = "second"
        d["dims"] = d["gemini_embedding"].astype(str).str.replace("d", ""); d["fusion"] = ""; d["seq"] = ""
    d["track"] = track
    return d.dropna(subset=["f1"])

def main():
    ap = argparse.ArgumentParser()
    for k in ["detection_linear","detection_deep","recovery_linear","recovery_deep","hier"]: ap.add_argument(f"--{k}")
    ap.add_argument("--out", default="results_by_track.csv")
    a = ap.parse_args()
    D = pd.concat([prep(a.detection_linear, False, "detection"), prep(a.detection_deep, True, "detection"),
                   prep(a.recovery_linear, False, "recovery"), prep(a.recovery_deep, True, "recovery")], ignore_index=True)
    rows = []
    def add(sub, track, task, fam, inputs, note=""):
        if not len(sub): return
        r = sub.loc[sub.f1.idxmax()]; f1f = folds(r, "sec_test_f1"); g = has_g(r.modality)
        aucf = folds(r, "sec_test_auc") if r.auc_unit == "second" else folds(r, "test_auc")
        rows.append(dict(track=track, task=task, model_family=fam, inputs=inputs, modality=r.modality,
            model=r.mtype, feature_set=r.feature_set, dataset=r.get("dataset",""), gemini_source=r.gemini_source if g else "",
            gemini_dims=r.dims if g else "", window=r.window, stride=r.stride, aggregation=r.aggregation, fusion=r.fusion, sequence_length=r.seq,
            F1=round(r.f1,3), F1_fold_std=round(f1f.std(ddof=1),3) if len(f1f)>1 else "", F1_ci_low=round(r.avg_sec_test_f1_ci_low,3),
            F1_ci_high=round(r.avg_sec_test_f1_ci_high,3), AUC=round(r.auc,3) if pd.notna(r.auc) else "", AUC_unit=r.auc_unit,
            AUC_fold_std=round(aucf.std(ddof=1),3) if len(aucf)>1 else "", FNR=round(r.avg_sec_test_fnr,3),
            Accuracy=round(r.avg_sec_test_accuracy,3), Precision=round(r.avg_sec_test_precision,3), Recall=round(r.avg_sec_test_recall,3),
            F1_per_fold=";".join(f"{x:.3f}" for x in f1f), n_runs_in_condition=len(sub), median_F1_in_condition=round(sub.f1.median(),3),
            top5_mean_F1=round(sub.f1.nlargest(5).mean(),3), run_name=r.Name, sweep=r.Sweep, note=note))
    for track in ["detection","recovery"]:
        for task in ["binary","multiclass"]:
            for fam in ["window-level","deep"]:
                s = D[(D.track==track)&(D.task==task)&(D.family==fam)]
                for m in CONV: add(s[s.modality==m], track, task, fam, label(m))
                for m in GEM:
                    for src in ["visual","visual_audio"]:
                        add(s[(s.modality==m)&(s.gemini_source==src)], track, task, fam, f"{label(m)} [{src}]")
                add(s[~s.modality.map(has_g)], track, task, fam, "BEST conventional (any)")
                add(s[s.modality=="gemini"], track, task, fam, "BEST Gemini-only (any source)")
                add(s[s.modality.map(has_g)], track, task, fam, "BEST Gemini-inclusive (any)")
            s = D[(D.track==track)&(D.task==task)]
            add(s[~s.modality.map(has_g)], track, task, "pooled", "BEST conventional (both families)")
            add(s[s.modality.map(has_g)], track, task, "pooled", "BEST Gemini-inclusive (both families)")
    out = pd.DataFrame(rows)
    if a.hier:
        h = pd.read_csv(a.hier); h = h[h.State=="finished"]
        hr = []
        def addh(sub, fam, inputs, note):
            if not len(sub): return
            r = sub.loc[sub.avg_sec_test_f1.idxmax()]
            hr.append(dict(track="escalation (v6) PROVISIONAL", task=r.binary_multiclass, model_family=fam, inputs=inputs, modality=r.modality,
                model=r.model_type if pd.notna(r.model_type) else f"L1={r.level1}/L2={r.level2}", feature_set=r.feature_set,
                gemini_source=r.gemini_source if has_g(r.modality) else "", gemini_dims=r.gemini_dims if has_g(r.modality) else "",
                window=r.window_sec, stride=r.stride_sec, F1=round(r.avg_sec_test_f1,3), F1_ci_low=round(r.avg_sec_test_f1_ci_low,3),
                F1_ci_high=round(r.avg_sec_test_f1_ci_high,3), AUC=round(r.avg_sec_test_auc,3), AUC_unit="second", FNR=round(r.avg_sec_test_fnr,3),
                n_runs_in_condition=len(sub), median_F1_in_condition=round(sub.avg_sec_test_f1.median(),3), run_name=r.Name, sweep=r.Sweep, note=note))
        hi = h[h.level2.notna()]; es = h[h.model_type.notna()]
        addh(hi[hi.level2=="none"], "hier", "flat 1 Hz (level2=none)", "control")
        for l2 in ["gru","lstm","rnn"]: addh(hi[hi.level2==l2], "hier", f"session RNN (level2={l2})", "requires noise-input control")
        addh(es[(es.use_current==True)&(es.use_delta_base!=True)&(es.use_delta_prev!=True)&(es.use_drift!=True)], "escalation", "current only", "control")
        addh(es[(es.use_delta_base!=True)&((es.use_delta_prev==True)|(es.use_drift==True))], "escalation", "relative, no baseline", "")
        addh(es[es.use_delta_base==True], "escalation", "with own-baseline calibration", "per-user calibration")
        out = pd.concat([out, pd.DataFrame(hr)], ignore_index=True)
    out.to_csv(a.out, index=False); print(out.shape, "->", a.out)

if __name__ == "__main__":
    main()
