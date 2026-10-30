# HRI'27 rebuttal runs (Oct 2026)

Every number quoted in the rebuttal is produced by a script in this repo and
written here. Items refer to `HRI27_rebuttal_triage.xlsx`.

| item | script | job | output |
|---|---|---|---|
| T2 temporal controls, T1 matched ladder, T11 seconds-based baseline, T10 relative 4x4 | `training/hierarchical_escalation/rebuttal_controls.py` | `rebuttal_controls.sub` (CPU, ~15 min) | `relative_controls.csv` |
| T1 / T11 as-swept tables | `training/analysis/pull_v6_escalation_export.py` | login node, ~5 min | `v6_escalation_export.csv` (regenerated, not in git) |
| T5 exact best recovery model, T10 absolute stage model, T3 SMOTE off (5 configs) | `training/analysis/run_headline.py --only 10..16` | `headline_rebuttal_A.sub`, `headline_rebuttal_B.sub` (CPU, 128 GB) | `headline_predictions_rebuttal_{A,B}.csv` (regenerated, not in git) |
| T10 second-level 4x4 for the absolute model | `training/analysis/confusion_second_level.py` | login node | `confusion_stage_absolute.csv` |
| T5 delay at k = 2, 3, 4 on the exact best model | `training/analysis/recovery_decay.py` (called by the report) | login node | `decay_best_exact_k{2,3,4}.csv` |
| everything, as markdown | `training/analysis/rebuttal_report.py` | login node | `REBUTTAL_RESULTS.md` |

Code changes made for the rebuttal, all additive (defaults reproduce the published pipeline):

* `use_smote` config switch in `temp_allscripts/{rf,sgd}_model.py` and `recovery/{rf,sgd}_model.py`;
* headline configs 10–16 in `run_headline.py` (exact reproductions of splendid-sweep-136 and
  true-sweep-651, and their SMOTE-off twins);
* new scripts listed above. Nothing in `escalation_linear.py`, `hier_data.py` or the published
  registries was modified.

Environment: `/share/ju/conda_virtualenvs/badidea` (the `badidea` conda env), as in every `.sub`.
