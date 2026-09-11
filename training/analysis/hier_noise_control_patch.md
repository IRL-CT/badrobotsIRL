# Patch: noise-input and random-onset controls for the v6 hierarchical / escalation models

## A. Noise-input control (`training/hierarchical_escalation/hier_model.py`)  — REQUIRED before any v6 number is used
Why: with level-2 recurrence the model sees the whole session from t = 0 in fixed windows; its hidden state
can count windows, and in this protocol the label is nearly a function of the count. If Gaussian noise with
identical session structure reproduces the score, the score is the schedule, not the person.

1. Add to the argparse / sweep config:
       "noise_input": {"values": [False, True]}
2. In `train()` (or wherever `sessions = H.build_sessions(...)` is called), immediately after building the
   train/val/test session lists and BEFORE `pad_sessions`:
       if cfg.noise_input:
           rng = np.random.default_rng(1000 + fold)
           for s in train_sessions + val_sessions + test_sessions:
               s["X"] = rng.standard_normal(s["X"].shape).astype("float32")
   (Keep `y`, `widx`, `part` untouched — the point is to keep every structural cue.)
3. Log `noise_input` in wandb.config so make_results_tables can separate the rows.
4. Sweep (small, CPU/GPU minutes): level2 ∈ {none, gru, lstm}, level1 ∈ {mean, gru}, window_sec ∈ {1, 3},
   stride_sec = 1, modality = facial (any), 3 seeds via `--seed` → 36 runs.
Decision rule (write it in Supp F verbatim): if noise-input level2≠none reaches second-level F1 ≥ 0.85, the
recurrent hierarchy is reported only as a demonstration of the protocol confound.

## B. Same control for the escalation-feature arm (`escalation_linear.py`) — already exists in spirit
The module docstring quotes a noise floor (AUC 0.55–0.61) but it was computed ad hoc. Add the same
`noise_input` flag (replace `session["X"]` before `escalation_features`) and run: use_current ∈ {T, F},
use_delta_base ∈ {T, F}, use_drift ∈ {T, F}, window_sec ∈ {1, 3, 5}, model_type = logreg, 3 seeds → ~72 runs, minutes.

## C. Random-onset variant (`hier_data.build_sessions`) — only if A passes
Why: a deployed robot does not know when the "session" began; removing the shared origin removes the clock
while keeping change-relative-to-recent-past features meaningful.
    def build_sessions(Xs, y, parts, window_sec, stride_sec, random_onset_max_s=0, seed=0):
        rng = np.random.default_rng(seed)
        ...
        for p in pd.unique(parts):
            m = parts == p; Xp, yp = Xs[m], y[m]
            if random_onset_max_s:
                cut = int(rng.integers(0, random_onset_max_s + 1)); Xp, yp = Xp[cut:], yp[cut:]
            ...
Apply with `random_onset_max_s=15` to train, val AND test sessions (fresh draw per fold). In
`escalation_linear.py`, `delta_base` is then computed from the first `baseline_sec` *observed* seconds.
Sweep: top-10 configs of each arm from the v6 export, 3 seeds. Report alongside the fixed-onset numbers.
