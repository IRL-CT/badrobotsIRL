# analysis_results

Uncapped W&B export for v4-v6 and the tables built from it.
Generated 2026-09-13. Every number here was measured on this machine.

## Why this exists

The exports in `corrections/` (`wandb_*_linear.csv`, `wandb_*_deep.csv`) are each
**exactly 1000 rows** -- W&B's default UI export cap, not the run count. Any
"best config" drawn from them is best-of-1000, not best-of-all. The real totals:

| project | runs exported | finished |
|---|---|---|
| brirl_gemini_v4_linear | 5,836 | 5,328 |
| brirl_gemini_v4_deep | 515 | 392 |
| brirl_gemini_v5_recovery_linear | 7,174 | 6,871 |
| brirl_gemini_v5_recovery_deep | 951 | 417 |
| brirl_hier_escalation_v6 | 12,257 | 12,245 |
| **total** | **26,733** | **25,253** |

## Files

| file | what it is |
|---|---|
| `raw/<project>.csv` | one row per W&B run, full config + summary, with run id and URL |
| `all_runs_clean.csv` | the four v4/v5 projects pooled, deep runs time-filtered, columns normalised across families (12,436 runs) |
| `ablation_v4_detection_binary.csv` | ablation table, 14 input cells |
| `ablation_v4_detection_multiclass.csv` | " |
| `ablation_v5_recovery_binary.csv` | " |
| `ablation_v5_recovery_multiclass.csv` | " |
| `v6_validated_summary.csv` | per-arm noise-control verdict for v6 |
| `v6_best_runs_by_arm.csv` | best real run per v6 arm, with `usable_in_paper` |

Rebuild with:

```
python3 training/analysis/export_all_wandb.py --entity mteresaparreira \
    --projects brirl_gemini_v4_linear brirl_gemini_v4_deep \
               brirl_gemini_v5_recovery_linear brirl_gemini_v5_recovery_deep \
               brirl_hier_escalation_v6 --out analysis_results/raw --skip-existing
python3 training/analysis/make_ablation_tables.py --raw analysis_results/raw --out analysis_results
python3 training/analysis/make_v6_summary.py     --raw analysis_results/raw --out analysis_results
```

## The deep time filter

Deep runs created before **2026-09-11T04:28:11Z** are dropped (168 detection,
206 recovery). That timestamp is when the post-Bug-A/B jobs started, and it
separates the two regimes exactly -- verified against the configs, not assumed:

| | before cutoff | after cutoff |
|---|---|---|
| `sequence_length` | 5 … **300** (Bug A) | **{5,10,15,30}** (the guard's cap) |
| `gemini_dims` | up to **3072** | <= 768 |
| second-level AUC exactly 0.500 | **32 of 94** (Bug B) | 2 of 158 |

Linear runs are NOT filtered: neither bug can reach them (sklearn
`predict_proba` returns two columns for binary, and no sequence construction
happens). A further 122 post-cutoff v5 deep runs carry
`status: skipped_invalid_combination` and no metrics -- these are the guard's
deliberate early exits, not failures, and are dropped with everything else that
has no second-level F1.

## Conventions (inherited from corrections/make_results_tables.py)

* the reported metric is **second-level** (`avg_sec_test_*`), not frame-level;
* deep AUC uses `avg_sec_test_auc` unless it is the 0.5 placeholder left by
  Bug B, in which case it falls back to frame-level and `AUC_unit` says `frame`.
  In the current tables all 56 cells are `second` -- no fallbacks survive;
* one row per input cell = the best finished run in that cell by second-level
  macro-F1, plus the spread over the whole cell (`n_runs`, `median_F1_in_cell`,
  `top5_mean_F1`).

## Reading a cell

`model_family` / `model` describe the run that WON the cell, not the cell.
Three cells are decided by a margin too small to mean anything, and say so in
their `note` column -- check `margin_over_other_family` before writing "deep
beats window-level" anywhere:

| table | cell | winner | other family | margin |
|---|---|---|---|---|
| v4 detection binary | P | deep/gru 0.724 | window-level 0.724 | **+0.00002** |
| v5 recovery binary | P | deep/transformer 0.688 | window-level 0.684 | +0.00411 |
| v5 recovery binary | best engineered | window-level/rf 0.731 | deep 0.730 | +0.00115 |

## v6: validated means the noise control passed

Not "scored well". The control replaces features with Gaussian noise and leaves
session structure identical; an arm fails if noise runs reach F1 >= 0.85.

| arm | real best | noise max | real - noise | verdict |
|---|---|---|---|---|
| session RNN (level2=rnn) | 0.979 | 0.986 | **-0.007** | FAILED |
| session LSTM | 0.974 | 0.974 | **0.000** | FAILED |
| session GRU | 0.966 | 0.967 | **-0.001** | FAILED |
| escalation features | 0.961 | 0.703 | **+0.258** | VALIDATED |
| flat 1 Hz (level2=none) | 0.808 | 0.560 | **+0.248** | VALIDATED |

All three recurrent arms are indistinguishable from noise: with level-2
recurrence the hidden state can count windows, and in a fixed protocol the count
determines the label. **No v6 recurrent number belongs in the main text.**

Two notes on how this differs from the earlier verdict in `WORK_REPORT.md` §3:

* it is computed from the sweep's own ~3k noise runs rather than the 72- and
  42-run dedicated grids, so it validates the configurations actually being
  reported (the grids pin modality `facial`/`full`/128d and window_sec {1,3};
  the best runs use other modalities and window_sec 5);
* it covers **`level2=rnn`**, a 563-run arm the dedicated grid never tested and
  which holds the highest raw F1 in all of v6. It fails.

The dedicated grids agree with the sweep on every arm they share.

### Still open for v6

The noise control asks whether the FEATURES matter. It does not address the
elapsed-time confound (`WORK_REPORT.md` §2/§2b), which is a separate question
and has not been run for v6. Treat the two validated v6 numbers as provisional
until a time-matched evaluation exists for them.
