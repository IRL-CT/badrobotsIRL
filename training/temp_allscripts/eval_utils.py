"""Evaluation at the true independent unit, with participant-level bootstrap CIs.

Why this module exists
----------------------
The dataset is stored at 100 rows per second of original video. Within a single
second the rows are near-duplicates (and for the Gemini embeddings they are
*exact* duplicates, since one embedding covers a whole second). A participant
contributing ~4000 rows therefore contributes only ~40 independent observations.

Frame-level metrics computed over ~19,000 test rows implicitly claim ~19,000
independent samples when the real number is closer to 200 (about 5 test
participants x ~40 seconds). Three consequences:

  1. Standard deviations come out roughly an order of magnitude too small, so
     models that are statistically indistinguishable look cleanly separated.
  2. Long label segments dominate the score purely because they contain more
     rows (Error III averages 7.8 s vs Error I's 5.3 s), not because they are
     more important.
  3. Fold-to-fold spread -- the honest source of uncertainty at N=24 -- is
     invisible next to the artificially tight per-fold standard deviation.

This module fixes both halves of that:

  * `aggregate_blocks` collapses consecutive rows back to one prediction per
    second of original video (majority vote for labels, mean for probabilities).
  * `participant_bootstrap` resamples *participants* with replacement to give a
    confidence interval at the unit that actually varies.

`get_all_metrics` here is a drop-in replacement for `common/get_all_metrics.py`.
It returns everything the original returned (unchanged keys, so existing
dashboards keep working) plus `sec_*` second-level metrics and their bootstrap
CIs.

Rows per second
---------------
`apply_windowing(df, window, stride, agg)` advances consecutive output rows by
`stride` original frames, so one second of video spans `100 / stride` model
rows. The sequence models slide a further window of `sequence_length` with step
1 over those rows, which does not change the spacing. The effective dataset
stride is `min(config.stride, config.window)`, matching `win_stride` in the
training scripts.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from get_all_metrics import get_all_metrics as _frame_level_metrics

BASE_FPS = 100
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

# Metrics computed at the second level and bootstrapped.
_METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "auc", "fnr")


# ======================================================================
# Unit conversion
# ======================================================================

def rows_per_second(dataset_stride, base_fps=BASE_FPS):
    """Number of consecutive model rows spanning one second of original video."""
    s = max(1, int(dataset_stride))
    return max(1, int(round(base_fps / s)))


def _dataset_stride_from_config():
    """Read the effective dataset stride from the active wandb config.

    Falls back to 1 (i.e. 100 rows per second) when no sweep config is
    available, which is the conservative choice: it aggregates the most.
    """
    try:
        import wandb
        config = wandb.config
        stride = int(config["stride"])
        window = int(config["window"])
        return max(1, min(stride, window))
    except Exception:
        return 1


# ======================================================================
# Aggregation to independent units
# ======================================================================

def _contiguous_runs(sessions):
    """Yield (start, stop) index pairs for each contiguous run of one session.

    `create_data_splits` selects test rows with a boolean mask over the source
    DataFrame, so rows stay in temporal order and each participant occupies one
    contiguous block. Runs are detected rather than using `np.unique` so that
    temporal order is never silently reordered.
    """
    if len(sessions) == 0:
        return
    boundaries = np.flatnonzero(sessions[1:] != sessions[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(sessions)]])
    for start, stop in zip(starts, stops):
        yield int(start), int(stop)


def aggregate_blocks(y_pred, y_true, sessions, y_proba=None, rows_per_unit=1):
    """Collapse consecutive rows within each session into blocks.

    Labels and predictions are reduced by majority vote; probabilities by mean
    (then renormalised). A trailing remainder is kept when it covers at least
    half a block, matching the convention in `common/get_all_metrics.py`.

    Returns
    -------
    (unit_pred, unit_true, unit_sessions, unit_proba)
        `unit_proba` is None when `y_proba` is None.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    sessions = np.asarray(sessions)
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 1:
            y_proba = y_proba.reshape(-1, 1)

    k = max(1, int(rows_per_unit))
    if k == 1:
        return y_pred, y_true, sessions, y_proba

    unit_pred, unit_true, unit_sessions, unit_proba = [], [], [], []

    for start, stop in _contiguous_runs(sessions):
        n = stop - start
        label = sessions[start]
        offset = 0
        while offset < n:
            remaining = n - offset
            if remaining < k and remaining < k / 2:
                break  # too short to count as a unit
            lo = start + offset
            hi = start + min(offset + k, n)
            unit_pred.append(stats.mode(y_pred[lo:hi], keepdims=False).mode)
            unit_true.append(stats.mode(y_true[lo:hi], keepdims=False).mode)
            unit_sessions.append(label)
            if y_proba is not None:
                unit_proba.append(y_proba[lo:hi].mean(axis=0))
            offset += k

    if len(unit_true) == 0:
        return y_pred, y_true, sessions, y_proba

    out_proba = None
    if y_proba is not None:
        out_proba = np.vstack(unit_proba)
        totals = out_proba.sum(axis=1, keepdims=True)
        out_proba = np.divide(
            out_proba, totals,
            out=np.full_like(out_proba, 1.0 / out_proba.shape[1]),
            where=totals > 0,
        )

    return (
        np.asarray(unit_pred),
        np.asarray(unit_true),
        np.asarray(unit_sessions),
        out_proba,
    )


# ======================================================================
# Core metrics
# ======================================================================

def _core_metrics(y_pred, y_true, y_proba=None):
    """Accuracy, macro precision/recall/F1, AUC and FNR on already-aggregated data."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = set(classes.tolist()).issubset({0, 1})

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=classes,
                                     average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=classes,
                               average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=classes,
                       average="macro", zero_division=0),
    }

    # AUC
    out["auc"] = float("nan")
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if is_binary:
                proba_pos = y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba.ravel()
                out["auc"] = roc_auc_score(y_true, proba_pos)
            else:
                # ovr macro needs every column to correspond to a present class
                present = np.unique(y_true)
                if y_proba.shape[1] == len(present):
                    out["auc"] = roc_auc_score(y_true, y_proba,
                                               multi_class="ovr", average="macro")
                else:
                    sub = y_proba[:, present]
                    totals = sub.sum(axis=1, keepdims=True)
                    sub = np.divide(sub, totals,
                                    out=np.full_like(sub, 1.0 / sub.shape[1]),
                                    where=totals > 0)
                    out["auc"] = roc_auc_score(y_true, sub,
                                               multi_class="ovr", average="macro",
                                               labels=present)
        except ValueError:
            out["auc"] = float("nan")

    # FNR: positive class for binary, macro-averaged otherwise
    if is_binary:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tp, fn = cm[1, 1], cm[1, 0]
        out["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        per_class = []
        for i in range(len(classes)):
            tp_c = cm[i, i]
            fn_c = cm[i, :].sum() - tp_c
            per_class.append(fn_c / (fn_c + tp_c) if (fn_c + tp_c) > 0 else 0.0)
        out["fnr"] = float(np.mean(per_class))

    return out


# ======================================================================
# Participant-level bootstrap
# ======================================================================

def participant_bootstrap(y_pred, y_true, sessions, y_proba=None,
                          n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Bootstrap the metrics by resampling participants with replacement.

    The participant is the unit that actually varies -- resampling rows would
    reproduce the same overconfidence this module exists to remove. With ~5 test
    participants per fold the interval is wide; that width is the point.

    Returns a dict of `<metric>_ci_low`, `<metric>_ci_high` and `<metric>_boot_std`.
    """
    sessions = np.asarray(sessions)
    unique_sessions = np.unique(sessions)
    index_by_session = {s: np.flatnonzero(sessions == s) for s in unique_sessions}

    rng = np.random.default_rng(seed)
    draws = {name: [] for name in _METRIC_NAMES}

    for _ in range(int(n_bootstrap)):
        picked = rng.choice(unique_sessions, size=len(unique_sessions), replace=True)
        idx = np.concatenate([index_by_session[s] for s in picked])
        if len(np.unique(y_true[idx])) < 2:
            continue  # degenerate resample: every unit one class
        m = _core_metrics(y_pred[idx], y_true[idx],
                          None if y_proba is None else y_proba[idx])
        for name in _METRIC_NAMES:
            draws[name].append(m[name])

    out = {}
    for name in _METRIC_NAMES:
        vals = np.asarray([v for v in draws[name] if not np.isnan(v)], dtype=float)
        if len(vals) == 0:
            out[f"{name}_ci_low"] = float("nan")
            out[f"{name}_ci_high"] = float("nan")
            out[f"{name}_boot_std"] = float("nan")
        else:
            out[f"{name}_ci_low"] = float(np.percentile(vals, CI_LOW_PCT))
            out[f"{name}_ci_high"] = float(np.percentile(vals, CI_HIGH_PCT))
            out[f"{name}_boot_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


# ======================================================================
# Drop-in replacement for common/get_all_metrics.get_all_metrics
# ======================================================================

def get_all_metrics(y_pred, y_true, y_pred_proba=None, sessions=None,
                    window_size=None, stride=None, tolerance=0,
                    dataset_stride=None, n_bootstrap=N_BOOTSTRAP):
    """Frame-level metrics (unchanged) plus second-level metrics and bootstrap CIs.

    Original keys (`test_f1`, `test_auc`, ...) keep their exact previous meaning
    so nothing downstream breaks. Added keys:

      sec_test_<metric>            metric at one prediction per second of video
      sec_test_<metric>_ci_low     2.5th percentile, participant bootstrap
      sec_test_<metric>_ci_high    97.5th percentile, participant bootstrap
      sec_test_<metric>_boot_std   bootstrap standard deviation
      eval_rows_per_second         rows collapsed into each unit
      n_eval_units                 independent units the sec_* metrics used
      n_eval_participants          participants in this fold's test set

    Compare `n_eval_units` against the frame count to see how much the
    frame-level numbers were overstating the sample size.
    """
    results = _frame_level_metrics(
        y_pred, y_true, y_pred_proba=y_pred_proba, sessions=sessions,
        window_size=window_size, stride=stride, tolerance=tolerance,
    )

    if sessions is None:
        return results

    effective_stride = (dataset_stride if dataset_stride is not None
                        else _dataset_stride_from_config())
    k = rows_per_second(effective_stride)

    unit_pred, unit_true, unit_sessions, unit_proba = aggregate_blocks(
        y_pred, y_true, sessions, y_proba=y_pred_proba, rows_per_unit=k,
    )

    results["eval_rows_per_second"] = k
    results["n_eval_units"] = int(len(unit_true))
    results["n_eval_participants"] = int(len(np.unique(unit_sessions)))

    for name, value in _core_metrics(unit_pred, unit_true, unit_proba).items():
        results[f"sec_test_{name}"] = value

    for key, value in participant_bootstrap(
        unit_pred, unit_true, unit_sessions, unit_proba,
        n_bootstrap=n_bootstrap,
    ).items():
        results[f"sec_test_{key}"] = value

    return results


# ======================================================================
# Session labels for sequence models
# ======================================================================

def sequence_sessions(sessions, sequence_length):
    """Session id for each sequence produced by `common.create_sequences`.

    `create_sequences` walks `np.unique(sessions)` and emits
    `n_s - sequence_length + 1` sequences for every session long enough,
    skipping shorter ones entirely. So it drops `sequence_length - 1` rows from
    *each* session, not `n_sessions * (sequence_length - 1)` rows from the front
    of the array.

    The training scripts approximated this as `session_test[-len(y_pred):]`,
    which removes the right *number* of entries but takes them all off the
    front. Every prediction after the first participant then carries the wrong
    participant id -- harmless for pooled frame-level metrics, but it silently
    corrupts anything that groups by participant: the windowed metrics, earliest
    detection time, and the bootstrap in this module.

    Returns an array aligned 1:1 with the model's predictions.
    """
    sessions = np.asarray(sessions)
    out = []
    for session in np.unique(sessions):
        n = int(np.sum(sessions == session))
        if n >= sequence_length:
            out.append(np.full(n - sequence_length + 1, session))
    if not out:
        return np.empty(0, dtype=sessions.dtype)
    return np.concatenate(out)
