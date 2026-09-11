# HRI 2027 Resubmission Plan — v3

Supersedes v2. Companion files: `HRI27_results_by_track.csv`, `HRI27_coding_and_runs_v2.md`, and the `hri27_analysis/` scripts. (`HRI27_results_analysis_v1.md` is an internal working note; nothing in it about earlier submissions or "corrections" appears in the paper.)

**Framing rule.** The only published prior work is Liu et al. (HRI'25 LBR: behavioural study; NERC'25: per-user models). This paper is a new cross-user study on that dataset. Design choices (silencing robot speech before audio embedding; separating the post-acknowledgment block; second-level metrics) are presented as methodology, never as fixes. Claims are scoped to *this scenario*: one robot, one scripted conversational failure repeated three times, 24 participants, in-lab Wizard-of-Oz.

**Narrative.** In this scenario, cross-user models detect *that* a conversational breakdown is under way with moderate reliability (macro-F1 ≈ 0.75–0.79, AUC ≈ 0.85), and they distinguish the post-failure state from the pre-failure baseline almost perfectly — but they largely fail to separate the aftermath of a failure from the failure itself: 33–53 % of post-acknowledgment seconds are classified as ongoing error. Distinguishing *which* successive error is under way is not resolved by the present experiments. Pretrained multimodal embeddings and engineered behavioural features reach similar performance; the small advantage for embeddings is not statistically distinguishable at N = 24.

---

## Abstract (draft)

> Social robots that fail repeatedly need to recognise, from the person in front of them, that something has gone wrong, how far the breakdown has progressed, and whether their attempt to recover has landed. Human reactions to robot errors are a promising signal for this, but they are expressed in highly individual ways, and models trained on one person rarely transfer to another. We study cross-user generalisation of behavioural error detection in a successive-failure scenario in which a social robot fails to understand its user three times in a row before acknowledging them. Using the 24-participant dataset of Liu et al., we evaluate three tasks under participant-disjoint cross-validation: detecting that an error has occurred, identifying which of the three successive errors is under way, and detecting the post-acknowledgment recovery phase. We compare engineered multimodal features (facial action units and gaze, acoustics, body pose, utterance content) with embeddings from a large pretrained audio-visual model, across window-level and sequence classifiers, and report metrics at the level of the independent observation (one second of interaction) with participant-level confidence intervals. Cross-user error detection reaches a macro-F1 of 0.76–0.79 (AUC 0.81–0.85) with either representation; pretrained embeddings offer a small, statistically indistinguishable advantage over engineered features, with the participant's voice contributing to the audio-visual embedding beyond video alone. The recovery phase is almost never confused with the pre-failure baseline (≤ 1 %), yet a third to a half of recovery seconds are classified as ongoing error: after the robot acknowledges the user, the user's behaviour continues to resemble reaction to failure. Identifying the specific error stage remains substantially harder. We further characterise a structural property of fixed-protocol failure studies — the near-determinism of labels given elapsed time — and evaluate models within time-matched windows to separate behavioural from temporal cues. We discuss implications for the timing of robot repair strategies and for the design of HRI datasets intended for cross-user learning.

(≈ 300 words; trim to the CFP limit. Numbers to be refreshed from the final tables; the stage-task sentence to be sharpened once the corrected multiclass runs exist — see `HRI27_coding_and_runs_v2.md`.)

---

## Page budget (8 pages, refs excluded — verify)

| Section | Pages |
|---|---|
| Title, abstract, Fig. 1 | 0.75 |
| 1 Introduction | 1.0 |
| 2 Related work | 0.75 |
| 3 Successive-failure scenario and dataset | 0.75 |
| 4 Tasks, features, models, evaluation | 1.5 |
| 5 Results (5.3 recovery largest) | 2.0 |
| 6 Discussion & limitations | 1.0 |
| 7 Conclusion | 0.25 |

---

## Title candidates
- *Detecting Failure and Its Aftermath: Cross-User Recognition of Successive Social Robot Errors from Human Reactions*
- *Still Reacting: What Human Behaviour Reveals About Robot Failure and Recovery Across Users*
- Avoid "learnability gap" unless defined once as the within-user (Liu et al.) vs cross-user difference.

## 1. Introduction (1.0 p)
- Define robot failure (Honig & Oron-Gilad 2018; Brooks 2017; Tian & Oviatt 2021); our case: scripted, unconditional, repeated conversational failure — the participant's strategy cannot affect the outcome.
- Three questions a repairing robot needs answered from behaviour alone: is something wrong (detect); how far has it gone (stage → clarify / change strategy / hand off); has the repair landed (recovery → when to stop repairing). Liu et al. showed reactions escalate and are learnable within a person; cross-user transfer and the post-acknowledgment phase have not been examined.
- Why cross-user; why HRI datasets make it hard (N = 24; idiosyncratic expression).
- Contributions (each maps to a result):
  1. A cross-user evaluation of successive-error detection with participant-disjoint folds, second-level metrics and participant-bootstrap CIs, comparing engineered features and pretrained audio-visual embeddings across modality combinations and model families.
  2. Recovery-phase detection as a task, with the finding that post-acknowledgment behaviour is separable from baseline but largely not from ongoing failure.
  3. An analysis of the elapsed-time structure of fixed-protocol failure studies and a time-matched evaluation isolating behavioural signal.
  4. Full tables, per-fold values, and code for reproducibility.
- Do not open on Liu et al.'s figure; describe contributions on their own terms.

## 2. Related Work (0.75 p)
1. Failure detection from social signals (Stiber; Kontogiorgos; ERR@HRI 1.0/2.0; Bremers; REFLEX; REPAIR-Bench). All treat single errors or independent trials; this dataset is, to our knowledge, the only one with successive dependent conversational errors and stage labels.
2. Repair and recovery in dialogue systems and HRI (Bohus & Rudnicky; Honig; Esterwood & Robert "three strikes"; REPAIR-Bench Task 3; Patamia et al. repair-attempt detection). None models the *user's* recovery state.
3. Individual variation in expression and the cross-user gap (personalised affect models; Barrett; Hoemann & Gendron).
4. Pretrained representations for affect and behaviour (MARLIN, HiCMAE, CLIP-based pipelines, frozen multimodal backbones).

## 3. Scenario and Dataset (0.75 p)
- Protocol (≤ 6 sentences; cite Liu et al. for detail); Wizard-of-Oz; scripted, unconditional; IRB exempt #1609006604; debrief.
- Participants: 30 completed; 4 excluded by Liu et al. (no footage: P1, P15, P30; no interaction: P13); 2 further excluded here (state reason — needed from you) → 24 analysed. Demographics → Supp A.
- Session structure: survey → Error I → II → III → acknowledgment, with typical durations (survey ≈ 19 s; Error I 5.3 s, Error III 7.8 s; the post-acknowledgment block ≈ 23 % of frames). This is the backbone of Fig. 1.
- Behavioural summary from Liu et al. (escalating displays; response-time growth; abandonment by 2 after Error II and 7 after Error III of 26). **Add**: the post-acknowledgment phase has not been characterised; participants remain vocal and animated (our label analysis, §4.2).
- The finding that AU04/06/14/23 differ between Errors II–III but not I–II motivates why stage identification is hard.

## 4. Tasks, Features, Models, Evaluation (1.5 p)
**4.1 Tasks.** Inputs: the participant's behaviour over a causal window; the robot's own speech is excluded from every input. Detection: pre-failure vs error, with the post-acknowledgment block excluded from both classes (its status is the subject of Task 3). Stage: pre-failure / I / II / III. Recovery: error vs recovery, and pre-failure / error / recovery. Independent unit: one second of one participant's interaction (24 participants, ≈ 40 s each, 72 error events).
**4.2 Label design.** Why the post-acknowledgment block is treated separately: on per-second embeddings, pre-failure vs post-acknowledgment seconds are more separable cross-user (AUC 0.875, logistic regression, leave-one-participant-out) than error vs non-error (0.680) — the two "no-error" phases are behaviourally distinct states. Class balances per scheme.
**4.3 Features.** OpenFace AUs + gaze (43); openSMILE on participant turns (25; robot turns removed via diarization); OpenPose upper body (24); CLIP text (512, PCA) + inter-utterance cosine (1). Pretrained embeddings: Gemini Embedding 2 on 1-s clips (480p/30 fps) with **the robot's utterances silenced in the audio track before embedding**, computed from video only and from video+audio at 128/256/768/3072-D, L2-normalised then z-scored, with per-second uniqueness checks. Feature-selected sets (RF importance, SelectKBest, domain-curated). Full lists → Supp B.
**4.4 Models.** Window-level classifiers (RF, SGD/linear-SVM, SVM, MLP, k-NN) over windowed features (window 1–30 frames, stride 1–10, mean/mode/last aggregation); sequence models (GRU, LSTM, Transformer; early/intermediate/late fusion; context ≤ 3 s — see §4.5). Hierarchical session model → Supp F only if controls pass.
**4.5 Evaluation.** 5-fold participant-disjoint CV (60/20/20, fixed seed); hyperparameters selected on validation folds by random search (run counts per condition in Supp C); SMOTE on training folds only. All metrics at the second level (majority vote / mean probability over the model rows within a second), macro-F1 primary, AUC and FNR secondary, mean ± SD over folds and participant-bootstrap 95 % CI; paired fold tests for headline contrasts; best-of-sweep reported with the condition median and top-5 mean. For sequence models, every second of every test session receives a prediction (left-padded contexts), so all models are scored on the same seconds.
**4.6 Elapsed-time structure.** In a fixed protocol, the label is close to a deterministic function of elapsed time; a classifier given elapsed seconds alone reaches AUC ≈ X (from `elapsed_time_baseline.py`). No model receives time or position as input; we additionally report metrics within the time band in which both classes occur across participants (§5.4).

## 5. Results (2.0 p)
**5.1 Detection (0.5 p).** Fig. 2 top + text. Best engineered-feature model: RF on audio + utterance cosine, F1 0.762 [0.66, 0.85], AUC 0.81. Best embedding-inclusive: RF on pose + facial + text + 3072-D video+audio embedding, F1 0.789 [0.70, 0.87], AUC 0.85. Paired over folds: +0.027 ± 0.039, 4/5 folds, p = 0.20 (n.s.). Embedding alone 0.757 ≈ engineered features. Video+audio embeddings exceed video-only (Δ ≈ 0.05 best, medians 0.69 vs 0.66; n.s.). Single modalities 0.72–0.74 with overlapping CIs; no modality dominates. Window-level RF is the strongest family; sequence models offer no advantage, and performance does not increase with context length. Median-of-sweep values (0.66–0.69) show best-of-sweep optimism of ≈ 0.1. Full table → Supp D.
**5.2 Stage (0.25 p).** Report only after the corrected runs (`HRI27_coding_and_runs_v2.md` P0-4/P0-5). Provisional wording: "In the present experiments, four-class stage identification did not exceed macro-F1 ≈ 0.4 (majority baseline 0.17) for any representation; confusion concentrates between Errors I and II (verify)." Table → Supp D.
**5.3 Recovery (0.9 p) — headline.** Fig. 3 (3-class confusion, best engineered | best embedding-inclusive, with per-class F1). Error vs recovery: F1 0.735 [0.62, 0.84], AUC 0.85; engineered 0.731 (Δ n.s.). Three-class: macro-F1 0.663 [0.57, 0.75], AUC 0.83; engineered 0.653. Recovery ↔ baseline confusion ≤ 1 %; recovery → error 33–53 %; recovery-class FNR 0.26–0.49. If `recovery_decay.py` runs: median seconds after acknowledgment until the model's prediction flips, and the share of participants for whom it never does within the block. Interpretation, scoped: in this scenario the robot's acknowledgment does not return the observed behaviour to baseline within the recorded interval; the post-failure state is distinct from baseline but continuous with failure.
**5.4 Elapsed-time structure (0.35 p).** Fig. 4: label-vs-time raster; AUC on all seconds vs within the time-matched band for elapsed-time-only, best engineered, best embedding models. State plainly what survives within the band.
**5.5 Session-level modelling (0–0.25 p; gated).** Only if noise-input and random-onset controls pass: report the gain of relative-to-own-baseline features as a *calibrated* setting (first 5 s of the same user). Otherwise one sentence in §6 and Supp F.

## 6. Discussion (1.0 p) — every paragraph tied to a result
- *Failure is legible; its resolution is less so.* Recovery result → repair-timing implication: a repairing robot should not wait for behavioural return-to-baseline as its confirmation signal within this time frame; acknowledgment first, then observe. Connect to abandonment (Liu et al.).
- *Representation is not the limiting factor here.* Embeddings ≈ engineered features; both plateau well below Liu et al.'s within-user 0.92. Participant diversity and protocol structure are the more plausible limits (learning curve if run; otherwise argued).
- *The audio channel.* Video+audio embeddings exceed video-only: the participant's own voice carries signal even when the robot's speech is removed — and removing it is necessary because the robot's utterances are the labels. (One sentence; a methodological recommendation, not a critique of anyone.)
- *Stage identification remains open* — behavioural similarity of Errors I and II; boundaries defined by interaction order; needs interaction-level context and datasets with variable timing.
- *Fixed protocols and the clock.* What such datasets can and cannot support; recommendation to randomise failure timing and count in future collections.
- *Generalisation to other embodiments / voice assistants*: acoustic-only ≈ 0.72 here.
- *Limitations*: single scenario, single failure type, one robot, in-lab WoZ; N = 24, five test participants per fold; offline evaluation; pretrained encoder is a black box with API cost; best-of-sweep optimism (medians reported); no open-weight encoder baseline (future work).
- *Ethics*: deception protocol, debrief, IRB.

## 7. Conclusion (0.25 p)
Three sentences mirroring the abstract's three findings and the two recommendations (repair timing; dataset design).

---

## Supplementary material (map)
A dataset & participants · B features & embedding pipeline · C sweeps, grids, run counts, selection protocol, metric definitions · D full results tables (from `HRI27_results_by_track.csv`) · E analyses (per-participant F1, paired tests, context-length sensitivity, feature-set comparison, video vs video+audio by dimensionality, learning curve if run) · F session-level models with noise-input and random-onset controls · Video: participant vignette with model predictions overlaid (consent permitting).

## Figures (implementation in `hri27_analysis/make_figures.py`; Fig. 1 described in `HRI27_results_analysis_v1.md` §7)
Fig. 1 teaser (session timeline + three tasks); Fig. 2 F1 dot-CI by input set; Fig. 3 recovery confusion; Fig. 4 elapsed-time structure; Fig. 5 per-participant (or Supp E).

## Reviewer-triage items resolved by the new design (for your tracking only)
T1 venue · T3 formulation (§4.1) · T4/T15 (no dominance claim to justify) · T5 (video vs video+audio) · T6 (definition, "scripted and unconditional") · T7 (Discussion claims ↔ results) · T8 (§4.6/§5.4) · T11 (participant funnel) · T12 (dataset positioning) · T13 (rewritten sentence) · S1 (design choice, §4.3) · S3 (curated sets in table) · S4 (medians/top-5) · S6/S7 (unit, CIs, paired tests) · S8 (family naming) · S9 (causal windows) · S14 (ethics) · S21 (term avoided). Open: T9 learning curve (P2), S2 feature audit (P2), S17 SMOTE (P2).
