# Coding changes and runs — v2 (everything runs on your side)

All analysis scripts are in `hri27_analysis/` — copy the folder to `training/analysis/` in the repo. Nothing below depends on me; every number in the paper is produced by a script in the repo from W&B exports or `runs_registry.csv`. P0 = required for an honest submission; P1 = needed for planned figures/tables; P2 = if time remains.

---

## Two pipeline problems found in the exports (fix before anything else)

**Bug A — sequence models drop the start of every session.** `common/create_data_splits.create_sequences` builds sequences from index 0 and assigns the label of the last element, so no prediction exists for the first `sequence_length` rows of each participant. With `sequence_length × effective_stride ≥ ~15 s` the entire pre-failure block vanishes from the test set (class-0 fraction 0.34 → 0.00 in the export). The three highest "stage" runs (F1 0.40–0.43) are 3-class on a truncated test set and are not valid; binary sequence results with long context are also on a different test set from the window-level models.
Fix (either): (i) left-pad each session with copies of its first row so every row gets a prediction — in `create_sequences`, `session_data = np.vstack([np.repeat(session_data[:1], sequence_length-1, 0), session_data])`, same for the corresponding frame indices; or (ii) cap `sequence_length × effective_stride ≤ 300 frames (3 s)` in every deep sweep grid. Recommend (i) plus the cap; then rerun the deep sweeps (P0-5).

**Bug B — second-level AUC missing for deep models.** `avg_sec_test_auc` = 0.5 in every deep run: `y_pred_proba` is not passed to `eval_utils.get_all_metrics`. Pass the sigmoid/softmax outputs. Fixed by the same rerun.

---

## Day 0 — exports, patches (no training)

**P0-1 Uncapped exports.**
```
python3 analysis/export_wandb.py --entity <ENT> --projects <detection_linear> <detection_deep> <recovery_linear> <recovery_deep> <hier> --out exports/
```
Check: `wc -l exports/*.csv` — none should be exactly 1000.

**P0-2 Registry patch** — follow `analysis/registry_patch.md` (adds `frame_idx`, `y_proba`, `track`, `label_scheme` to `runs_registry.csv`). Needed by every prediction-based analysis.

**P0-3 Bug A + Bug B patches** in `common/create_data_splits.py` and the six `*_wandb.py` deep scripts (temp_allscripts and recovery). Add `sequence_length × min(stride, window) ≤ 300` as a config guard.

**P0-4 Confirm the stage (multiclass) window-level sweep.** The export has no `binary_multiclass = multiclass` rows for detection. If the sweep ran, export it; if not, launch (P0-6).

---

## Day 1 — required runs

**P0-5 Deep reruns after Bug A/B** — detection binary, detection multiclass, recovery binary, recovery multiclass. Grid as v4/v5 but: `sequence_length ∈ {5, 10, 15, 30}` with the 3-s guard, `gemini_dims ≤ 768` (the 3072-D + long-sequence runs are the ones that failed/crashed: 333 of 547 recovery_deep runs), `--count 150` per (model × task). ≈ 12 sweeps × 150 runs; start on day 1 so they run while you write. Sequence-model rows in the paper come only from these.

**P0-6 Stage window-level sweep** (if missing): `temp_allscripts/{rf,sgd,mlp}_model.py` with `binary_multiclass` pinned to `multiclass`, rest of the v4 grid, `--count 250` each (skip k-NN/SVM). Confusion matrices are already logged per fold.

**P0-7 Headline configs re-run with the patched registry** (window-level, minutes each, CPU):

| # | Track / task | Config (from `HRI27_results_by_track.csv`) | run_name to reproduce |
|---|---|---|---|
| 1 | detection binary, engineered | RF · audio+cosine · rf set · win 1 · str 10 · mode | giddy-sweep-29 |
| 2 | detection binary, embedding-incl. | RF · pose+facial+text+gemini · full · visual_audio · 3072 · win 10 · str 3 · last | iconic-sweep-99 |
| 3 | detection binary, embedding-only | SGD · gemini · selectkbest · visual_audio · 256 · win 15 | (BEST Gemini-only row) |
| 4 | recovery binary, engineered | RF · pose+facial+audio · full · win 30 | fluent-sweep-21 |
| 5 | recovery binary, embedding-incl. | RF · pose+facial+audio+gemini · full · visual_audio · 128 · win 10 · str 10 · average | splendid-sweep-136 |
| 6 | recovery 3-class, engineered | RF · pose+facial+audio · full | volcanic-sweep-232 |
| 7 | recovery 3-class, embedding-incl. | RF · facial+audio+text+gemini · full · visual_audio · 768 | deep-sweep-133 |
Add `--no_smote` variants of 1, 2, 4, 5 if P2-2 is wanted. Record each `run_id`.

**P0-8 Elapsed-time baseline** (CPU, seconds):
```
cd training/hierarchical_escalation
python3 ../analysis/elapsed_time_baseline.py --scheme v4 --features elapsed --out_preds elapsed_v4.csv
python3 ../analysis/elapsed_time_baseline.py --scheme v4 --features elapsed,speaking --out_preds elapsed_speak_v4.csv
python3 ../analysis/elapsed_time_baseline.py --scheme v5_binary --features elapsed --out_preds elapsed_v5b.csv
```
The first number goes into §4.6.

**P0-9 Noise-input control for the hierarchical model** — `analysis/hier_noise_control_patch.md` §A (36 runs, minutes) and §B (≈ 72 runs). Decision rule is written in the patch. Until this runs, no v6 number is used anywhere.

---

## Day 2 — analyses (all CPU, from registry + exports)

**P1-1 Results table (Supp D; input to Fig. 2)**
```
python3 analysis/make_results_tables.py --detection_linear exports/<..>.csv --detection_deep exports/<..>.csv \
  --recovery_linear exports/<..>.csv --recovery_deep exports/<..>.csv --hier exports/<..>.csv --out results_by_track.csv
```

**P1-2 Time-matched evaluation (§5.4, Fig. 4 right)**
```
python3 analysis/time_matched_eval.py --registry runs_registry.csv --run_id <id1> --run_id <id2> --run_id <id3> \
  --elapsed_preds elapsed_v4.csv --out time_matched_detection.csv
python3 analysis/time_matched_eval.py --registry runs_registry.csv --run_id <id4> --run_id <id5> \
  --elapsed_preds elapsed_v5b.csv --out time_matched_recovery.csv
```

**P1-3 Per-participant F1 and paired tests (§5.1, §5.3, Fig. 5)**
```
python3 analysis/per_participant_and_paired.py --registry runs_registry.csv --run_a <id2> --run_b <id1> --label_a embedding --label_b engineered --out pp_detection.csv
python3 analysis/per_participant_and_paired.py --registry runs_registry.csv --run_a <id5> --run_b <id4> --label_a embedding --label_b engineered --out pp_recovery.csv
```

**P1-4 Recovery decay (§5.3)**
```
python3 analysis/recovery_decay.py --registry runs_registry.csv --run_id <id5> --k 3 --out decay_embedding.csv
python3 analysis/recovery_decay.py --registry runs_registry.csv --run_id <id4> --k 3 --out decay_engineered.csv
```

**P1-5 Figures**
```
python3 analysis/make_figures.py fig2 --results results_by_track.csv --out figs/
python3 analysis/make_figures.py fig3 --recovery_linear exports/<..>.csv --recovery_deep exports/<..>.csv --out figs/
python3 analysis/make_figures.py fig4 --base_csv data/interpolated/allparticipants_100fps.csv --time_matched time_matched_detection.csv --out figs/
python3 analysis/make_figures.py fig5 --per_participant pp_detection.csv --out figs/
```
Fig. 1 (teaser) is manual: three stacked axes sharing a time axis for one held-out participant — pose skeletons or blurred frames at fixed seconds with the five block labels; two or three 1-Hz feature traces with dashed lines at the four robot utterances; P(error) from run #2 and P(recovery) from run #5 (from `registry_to_seconds.load_run`) as filled curves with the three task brackets. Caption: inputs are participant behaviour only; robot speech is excluded.

**P1-6 Stage confusion (§5.2)** — after P0-5/P0-6: pooled 4×4 for best engineered and best embedding runs (same `_cms` helper as fig3; add a `fig3b` if wanted), adjacent-stage accuracy, mean absolute stage error.

---

## Day 3 — conditional and optional

**P1-7 Random-onset variant (v6)** — only if P0-9 shows the recurrent hierarchy does *not* score ≥ 0.85 on noise. `hier_noise_control_patch.md` §C; top-10 configs of each arm, 3 seeds.

**P2-1 Learning curve (T9)** — runs #1 and #2 trained on N ∈ {4, 8, 12, 16} participants (3 random subsets each, fixed test fold). Add `--train_participants N --subset_seed s` to `rf_model.py`; ≈ 24 runs, minutes.

**P2-2 SMOTE on/off** for runs #1, #2, #4, #5 (add `--no_smote`).

**P2-3 `verbal_response_time` audit (S2)** — check `curated_features_dataset_v5.csv` for features computed relative to error onset; if present, drop and rerun the `curated_v5` rows (RF only).

## Not in scope for this deadline
Open-weight encoder baseline (state as future work) · embedding-space analyses · multi-seed reruns of full sweeps · LOPO.

## Reporting rules (encode in tables and text)
- Unit: second-level. Frame-level only in Supp for reference to Liu et al.
- Every headline number: CI, per-fold values, n runs in condition, condition median and top-5 mean.
- Accuracy reported but not used for comparison (class priors differ across label schemes).
- Deep-model rows only from the post-fix reruns (P0-5); the current export's deep rows are not used.
- No v6 (hierarchical/escalation) numbers in the main text unless P0-9 (and P1-7) pass.
- Stage-task sentence written only after P0-5/P0-6.
