# Work report — response to the LLM revision plan

Covers everything done against `HRI27_coding_and_runs_v2.md` / `HRI27_revision_plan_v3.md`.
Written 2026-09-11. Numbers here were measured on this machine, not quoted from the plan.

---

## 1. Verdict on the plan's two claimed bugs

### Bug A — sequence models drop the start of every session — **CONFIRMED**

`common/create_data_splits.create_sequences` assigns the label of the **last**
element of each window:

```python
targets.append(session_target[i + sequence_length - 1])
```

so the first `(sequence_length - 1) x effective_stride` frames of every session
receive no prediction. Those frames are the beginning of the session — the
pre-failure block.

| seq_len | stride | seconds lost | vs the 10.4 s median pre-failure block |
|---|---|---|---|
| 15 | 10 | 1.4 s | fine |
| 100 | 10 | 9.9 s | almost all gone |
| 150 | 10 | **14.9 s** | **entirely gone** |
| 300 | 10 | **29.9 s** | most of a 39 s session |

Measured in `HRI27_results_by_track.csv`: **10 of 93 deep rows lost the whole
pre-failure block, 26 lost more than 30 %.** The damage lands exactly where the
plan said: **all three top detection-multiclass (stage) runs** are
`sequence_length = 300` runs losing 14.95 s, at F1 0.431 / 0.414 / 0.403 —
matching the plan's "F1 0.40–0.43". Those stage numbers were computed on a test
set containing **no class 0 at all** and are not usable.

### Bug B — second-level AUC missing for deep models — symptom real, **stated mechanism wrong**

The plan says `y_pred_proba` is not passed to `eval_utils.get_all_metrics`. It
**is** passed, at every deep call site. The plan also says `avg_sec_test_auc` is
0.5 "in every deep run"; in its own CSV, **0 of 93 deep rows are exactly 0.5**
(range 0.440–0.875).

Something was nevertheless broken: **55 of 93 deep rows fell back to frame-level
AUC.** The real cause was found by splitting binary from multiclass —
`tr_bin` gave exactly 0.5 in 4/4 runs, `tr_mc` gave real values in 40/40 — and it
is a bug introduced in `eval_utils.aggregate_blocks` during this project, not in
the original code:

A binary Keras head emits **one sigmoid column**. `aggregate_blocks` renormalised
probabilities by their row sum, which divides that single column by itself and
makes every probability exactly 1.0 → constant scores → AUC exactly 0.5.
Multiclass softmax (≥ 2 columns) was unaffected. Reproduced directly:

```
single-column sigmoid:  frame-level AUC 0.995  ->  second-level 0.500
two-column softmax:     unaffected
```

Fixed in all three `eval_utils.py` copies (renormalise only when `shape[1] > 1`).

---

## 2. The finding that outranks everything else

`analysis/elapsed_time_baseline.py`, logistic regression on **one feature** —
seconds since session start, no behavioural data whatsoever, same participant
folds as every sweep:

| label scheme | macro-F1 | AUC |
|---|---|---|
| v4 detection | **0.976 ± 0.023** | **0.999 ± 0.002** |
| v5 error-vs-recovery | **0.876 ± 0.051** | **0.959 ± 0.026** |

The best embedding model in the results table is F1 0.789 / AUC 0.851. **The
clock alone beats every model in the paper.**

What this is and is not:

* It is **not** leakage. No model receives time, frame index or position as an
  input. Splits are participant-disjoint and rows stay in temporal order; nothing
  is shuffled across time.
* It **is** a construct-validity confound. The protocol is fixed — survey, then
  Error I, II, III, always in that order at roughly the same times — so the label
  is very nearly a deterministic function of elapsed time. Anything that drifts
  across a session therefore predicts the label: rising arousal (the intended
  signal), but equally posture settling, lighting, autoexposure, fatigue.
* Consequence: a model at AUC 0.85 may be reading behaviour or may be reading
  "time has passed", and the aggregate number cannot distinguish them.

The remedy is already in the plan (§5.4): score only within the time band where
both classes occur across participants. That analysis is the one that converts
this from a threat into a result, and **no headline claim should be written
without it**.

---

## 3. Section A noise control — **FAILED**, v6 recurrence is a confound

72-run grid (`level2 x level1 x window_sec x noise_input x seed`), features
replaced by Gaussian noise with the session structure left identical.

median / max macro-F1:

| level 2 | real | **noise** |
|---|---|---|
| `none` (flat) | 0.638 / 0.682 | **0.481 / 0.522** |
| `gru` | 0.802 / 0.852 | **0.703 / 0.852** |
| `lstm` | 0.790 / 0.811 | **0.781 / 0.925** |

Decision rule from `hier_noise_control_patch.md`: *if noise-input runs with
`level2 != none` reach F1 ≥ 0.85, the recurrent hierarchy is reported only as a
demonstration of the protocol confound.*

**5 of 24 such runs reach ≥ 0.85; the best scores F1 0.925, AUC 0.992 on pure
noise.** The rule fails. Top offenders:

```
level1  level2  win  seed     F1     AUC
   gru    lstm    1     2  0.925   0.992
   gru    lstm    3     0  0.916   0.987
   gru    lstm    3     2  0.915   0.986
```

Read the `lstm` row of the table: real 0.790 vs noise 0.781 — a gap of **0.009**.
With level-2 recurrence the hidden state can simply count windows, and in this
protocol the count determines the label. The apparent performance of the
recurrent hierarchy is the schedule.

**One genuinely positive result sits inside this.** The flat control
(`level2=none`) shows real 0.638 vs noise 0.481 — a 0.157 gap. The per-window
model *is* reading behaviour. It is specifically the session-level recurrence
that is contaminated.

Section B (escalation-feature arm) **passes**: noise max F1 0.611 against the
0.85 threshold, real median 0.664 vs noise 0.472.

Implication: no v6 recurrent number in the main text. Report it as a confound
demonstration, or report the flat and escalation-feature arms only.

---

## 4. Code changes

| change | where |
|---|---|
| **Bug A cap**: runtime guard `sequence_length x min(stride, window) <= 300` (3 s) + grid capped to `[5,10,15,30]`; `gemini_dims` capped at 768 (3072-D + long sequences is what OOM-killed jobs) | 12 deep scripts, v4 + v5 |
| **Bug B fix**: renormalise probabilities only when `shape[1] > 1` | `eval_utils.py` x3 |
| **`--task {both,binary,multiclass}`** to pin the sweep to one task | 10 linear scripts |
| **`noise_input` and `seed`** as swept, logged conditions | both v6 arms |
| **`analysis/`** folder created from the plan's scripts; fixed a broken `sys.path` in `elapsed_time_baseline.py` (it built `training/analysis/../training/hierarchical_escalation`) | `training/analysis/` |
| **`analysis/run_headline.py`** — runs the 7 headline configs by importing the real model modules with a stubbed wandb config, intercepting `apply_windowing` (for `frame_idx`) and `get_all_metrics` (for per-fold predictions + probabilities). Writes `headline_predictions.csv` in the schema `registry_to_seconds.load_run` expects, so the 3.8 GB registries stay untouched | new |
| **`hierarchical_escalation/run_noise_control.py`** — resumable Section A driver | new |

### Launch-infrastructure bugs found and fixed

* `--get-user-env` propagates whatever `python3` the submitting shell had. Submitted
  without the conda env active, that is anaconda3, which has no keras — **all 12
  deep jobs died instantly** on `No module named 'keras'`. The interpreter is now
  pinned absolutely (`PYBIN=...`) in all 24 `.sub` files.
* `WANDB_SERVICE_WAIT` is **silently ignored** by wandb 0.16, which reads
  `WANDB__SERVICE_WAIT` (double underscore). Roughly 20 % of sweep runs were dying
  with `ServiceStartTimeoutError`. Corrected everywhere.
* Long analyses launched as login-node `nohup` were killed when the session ended.
  The Section A control lost nine hours this way and wrote nothing, because it only
  saved its CSV at the end. Both long analyses are now SLURM jobs with `--requeue`
  that **resume and append after every run**.

---

## 5. Things the plan assumed that turned out to be false

* **The detection-stage window-level sweep had never run.**
  `wandb_detection_linear.csv`: 1000 rows, **all `binary`, zero `multiclass`**.
  Both tasks shared one sweep, so random search starved the stage task. Now
  running pinned, `--task multiclass --count 250`, for **all five** model types.
* **Both linear exports are exactly 1000 rows** — W&B's default export cap, not
  the run count. Any "best config" drawn from them is best-of-1000, not
  best-of-all. Use `analysis/export_wandb.py` for uncapped exports before
  finalising anything.
* **`create_data_splits` 5-fold CV is not a clean partition.** `p19nodbot` appears
  in two test folds; 25 test slots over 24 participants. Pre-existing, affects
  every version, reproduced deliberately in v6 for comparability.

---

## 6. State at time of writing

**Complete**

* Section A noise control — 72/72, FAILED (see §3)
* Section B noise control — PASSED
* Elapsed-time baseline — both schemes (see §2)
* All code changes in §4

**Running (19 SLURM jobs)**

* v4 deep x6, v5 deep x5 — post-Bug-A/B reruns, `--count 150`
* v4 stage linear x5 (`knn_mc mlp_mc rf_mc sgd_mc svm_mc`), `--count 250`
* v6 `h_esc`, `h_hier`
* `headline` — on config 7 of 7; `headline_predictions.csv` already has configs 1–6

**Not started**

* **Time-matched evaluation (§5.4)** — the single most important remaining
  analysis, given §2. Needs `headline_predictions.csv`, which is nearly done.
* P1-3 per-participant / paired tests, P1-4 recovery decay, P1-5 figures
* P1-7 random-onset variant — the plan gates this on Section A passing. It
  **failed**, so this becomes more relevant, not less: removing the shared time
  origin is the obvious next test of whether anything survives.

## 7. Which results to discard

* **All deep runs from before 2026-09-11** — invalid under Bug A and Bug B.
  Projects `brirl_gemini_v4_deep`, `brirl_gemini_v5_recovery_deep`.
* **All v6 recurrent numbers** — Section A failed.
* **Keep all linear/window-level runs.** Neither bug touches them (sklearn
  `predict_proba` returns two columns for binary; no sequence construction).

W&B projects: `brirl_gemini_v4_linear`, `brirl_gemini_v4_deep`,
`brirl_gemini_v5_recovery_linear`, `brirl_gemini_v5_recovery_deep`,
`brirl_hier_escalation_v6`.
