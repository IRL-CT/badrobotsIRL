import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def get_all_metrics(y_pred, y_true, y_pred_proba=None, sessions=None,
                    window_size=None, stride=None, tolerance=0):
    """
    Compute extended evaluation metrics for frame-level predictions.

    Parameters
    ----------
    y_pred : array-like
        Predicted labels.
    y_true : array-like
        Ground-truth labels.
    y_pred_proba : array-like or None
        Prediction probabilities (n_samples, n_classes). Required for AUC.
    sessions : array-like or None
        Session/participant ID per sample. Required for windowed predictions
        and earliest detection time.
    window_size : int or None
        Fixed window length for windowed predictions (mode voting).
        Passed from the model training script.
    stride : int or None
        Step size between consecutive windows. Defaults to window_size
        (non-overlapping). Set stride < window_size for overlapping windows.
    tolerance : int
        Tolerance window for tolerant metrics (same as original get_metrics).

    Returns
    -------
    dict
        Dictionary with all computed metrics.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if sessions is not None:
        sessions = np.array(sessions)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = set(classes).issubset({0, 1})

    # ------------------------------------------------------------------
    # 1. Original strict & tolerant metrics
    # ------------------------------------------------------------------
    y_pred_tolerant = y_pred.copy()

    if tolerance > 0:
        for i in range(len(y_pred)):
            if sessions is not None:
                same_session_indices = np.where(sessions == sessions[i])[0]

                if i == same_session_indices[same_session_indices <= i][0]:
                    start = same_session_indices[same_session_indices <= i][0]
                else:
                    start = max(
                        same_session_indices[same_session_indices <= i][-1] - tolerance,
                        same_session_indices[same_session_indices <= i][0]
                    )

                if i == same_session_indices[same_session_indices >= i][-1]:
                    end = same_session_indices[same_session_indices >= i][-1]
                else:
                    end = min(
                        same_session_indices[same_session_indices >= i][0] + tolerance + 1,
                        same_session_indices[same_session_indices >= i][-1] + 1
                    )
            else:
                start = max(0, i - tolerance)
                end = min(len(y_true), i + tolerance + 1)

            if y_pred[i] in y_true[start:end]:
                y_pred_tolerant[i] = y_true[i]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    accuracy_tolerant = accuracy_score(y_true, y_pred_tolerant)
    precision_tolerant = precision_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)
    recall_tolerant = recall_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)
    f1_tolerant = f1_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)

    results = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_accuracy_tolerant": accuracy_tolerant,
        "test_precision_tolerant": precision_tolerant,
        "test_recall_tolerant": recall_tolerant,
        "test_f1_tolerant": f1_tolerant,
    }

    # ------------------------------------------------------------------
    # 2. AUC (Area Under the ROC Curve)
    # ------------------------------------------------------------------
    if y_pred_proba is not None:
        y_pred_proba = np.array(y_pred_proba)
        try:
            if is_binary:
                # Use probability of positive class (column 1)
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                    proba_pos = y_pred_proba[:, 1]
                elif y_pred_proba.ndim == 2:
                    # Binary sigmoid output: single column IS the positive-class prob
                    proba_pos = y_pred_proba.ravel()
                else:
                    proba_pos = y_pred_proba
                auc = roc_auc_score(y_true, proba_pos)
            else:
                auc = roc_auc_score(y_true, y_pred_proba,
                                    multi_class="ovr", average="macro")
            results["test_auc"] = auc
        except ValueError:
            # Happens when only one class is present in y_true
            results["test_auc"] = float("nan")
    else:
        results["test_auc"] = float("nan")

    # ------------------------------------------------------------------
    # 3. False Negative Rate  (FNR = FN / (FN + TP) = 1 - TPR)
    #    For binary: computed on the positive class (label 1).
    #    For multiclass: macro-averaged across all classes.
    # ------------------------------------------------------------------
    if is_binary:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tp = cm[1, 1]
        fn = cm[1, 0]
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        results["test_fnr"] = fnr
    else:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fnr_per_class = []
        for idx in range(len(classes)):
            tp_c = cm[idx, idx]
            fn_c = cm[idx, :].sum() - tp_c
            fnr_c = fn_c / (fn_c + tp_c) if (fn_c + tp_c) > 0 else 0.0
            fnr_per_class.append(fnr_c)
        results["test_fnr"] = float(np.mean(fnr_per_class))

    # ------------------------------------------------------------------
    # 4. Windowed Predictions (fixed-length, mode voting)
    #    Splits each session into windows of `window_size`
    #    frames.  The window-level prediction is the statistical mode of
    #    the per-frame predictions inside that window; likewise for labels.
    # ------------------------------------------------------------------
    if window_size is not None and window_size > 0 and sessions is not None:
        effective_stride = stride if stride is not None else window_size
        win_preds, win_trues = _windowed_predictions(
            y_pred, y_true, sessions, window_size, effective_stride
        )
        if len(win_preds) > 0:
            # Filter out NaN entries that can arise from stats.mode on
            # very small remainder windows
            valid_mask = ~(np.isnan(win_preds) | np.isnan(win_trues))
            win_preds = win_preds[valid_mask]
            win_trues = win_trues[valid_mask]

        if len(win_preds) > 0:
            win_classes = np.unique(np.concatenate([win_trues, win_preds]))
            results["test_windowed_accuracy"] = accuracy_score(win_trues, win_preds)
            results["test_windowed_precision"] = precision_score(
                win_trues, win_preds, labels=win_classes,
                average="macro", zero_division=0
            )
            results["test_windowed_recall"] = recall_score(
                win_trues, win_preds, labels=win_classes,
                average="macro", zero_division=0
            )
            results["test_windowed_f1"] = f1_score(
                win_trues, win_preds, labels=win_classes,
                average="macro", zero_division=0
            )
        else:
            results["test_windowed_accuracy"] = float("nan")
            results["test_windowed_precision"] = float("nan")
            results["test_windowed_recall"] = float("nan")
            results["test_windowed_f1"] = float("nan")
    else:
        results["test_windowed_accuracy"] = float("nan")
        results["test_windowed_precision"] = float("nan")
        results["test_windowed_recall"] = float("nan")
        results["test_windowed_f1"] = float("nan")

    # ------------------------------------------------------------------
    # 5. Earliest Detection Time (binary only)
    #    Per session: number of frames between the first true error
    #    (y_true == 1) and the first correct detection (y_pred == 1)
    #    at or after that point. Averaged across sessions that contain
    #    at least one true error.
    # ------------------------------------------------------------------
    if is_binary and sessions is not None:
        edt, edt_details = _earliest_detection_time(
            y_pred, y_true, sessions
        )
        results["test_earliest_detection_time"] = edt
        results["test_edt_per_session"] = edt_details
    else:
        results["test_earliest_detection_time"] = float("nan")
        results["test_edt_per_session"] = {}

    return results


# ======================================================================
# Helper functions
# ======================================================================

def _windowed_predictions(y_pred, y_true, sessions, window_size, stride):
    """Return window-level mode predictions and labels.

    Parameters
    ----------
    stride : int
        Step size between consecutive windows. When stride == window_size
        the windows are non-overlapping; when stride < window_size they
        overlap.
    """
    win_preds = []
    win_trues = []

    for session in np.unique(sessions):
        mask = sessions == session
        sess_pred = y_pred[mask]
        sess_true = y_true[mask]
        n_frames = len(sess_pred)

        start = 0
        while start + window_size <= n_frames:
            end = start + window_size
            win_preds.append(stats.mode(sess_pred[start:end], keepdims=False).mode)
            win_trues.append(stats.mode(sess_true[start:end], keepdims=False).mode)
            start += stride

        # Handle remaining frames if they form at least half a window
        remainder = n_frames - start
        if remainder >= window_size // 2:
            win_preds.append(stats.mode(sess_pred[start:], keepdims=False).mode)
            win_trues.append(stats.mode(sess_true[start:], keepdims=False).mode)

    return np.array(win_preds), np.array(win_trues)


def _earliest_detection_time(y_pred, y_true, sessions):
    """
    Compute earliest detection time per session.

    For each session that contains at least one true error (label 1),
    find:
      - first_error_idx : index of the first frame where y_true == 1
      - first_detect_idx: first frame at or after first_error_idx where
                          y_pred == 1

    EDT for that session = first_detect_idx - first_error_idx  (in frames).
    If the error is never detected, EDT = number of remaining frames
    (i.e. worst case).

    Returns
    -------
    mean_edt : float
        Average EDT across sessions with errors. NaN if no session has errors.
    details : dict
        Per-session EDT values.
    """
    details = {}

    for session in np.unique(sessions):
        mask = sessions == session
        sess_pred = y_pred[mask]
        sess_true = y_true[mask]

        error_indices = np.where(sess_true == 1)[0]
        if len(error_indices) == 0:
            continue  # no errors in this session

        first_error = error_indices[0]
        detections_after = np.where(sess_pred[first_error:] == 1)[0]

        if len(detections_after) > 0:
            edt = int(detections_after[0])  # relative to first_error
        else:
            # Never detected: worst-case = remaining frames after first error
            edt = int(len(sess_true) - first_error)

        details[session] = edt

    if len(details) == 0:
        return float("nan"), details

    mean_edt = float(np.mean(list(details.values())))
    return mean_edt, details
