# Patch: per-fold predictions with time index and probabilities

All prediction-based analyses (time-matched evaluation, per-participant F1, recovery decay, Figs. 1/4/5)
read `runs_registry.csv`. It currently stores y_pred / y_true / session per fold but not *when* each row
occurred or the class probabilities. Two edits, then re-run the headline configs listed in the run plan.

## 1. `training/common/registry_utils.py` (and the copy in recovery/)
Add to REGISTRY_COLS after "session":
    "frame_idx",   # JSON list: first ORIGINAL 100 fps frame index of each evaluated row (within participant)
    "y_proba",     # JSON list: P(class) per row; for binary a float, for multiclass a list of floats
    "track",       # 'detection' | 'recovery'
    "label_scheme" # 'v4' | 'v5_binary' | 'v5_multiclass'
and write them in append_to_registry:
    "frame_idx": _to_json_list(fold_df["frame_idx"].values),
    "y_proba":   json.dumps(np.asarray(fold_df["y_proba"].tolist()).tolist()),
    "track": config.get("track", ""), "label_scheme": config.get("label_scheme", ""),

## 2. Each model script (knn/mlp/rf/sgd/svm_model.py, gru/lstm/transformer_*_wandb.py)
Where the per-fold DataFrame is built (columns fold, y_pred, y_true, session), add:
    fold_df["frame_idx"] = test_frame_idx      # see below
    fold_df["y_proba"]   = list(proba)         # sklearn: clf.predict_proba(X_test)[:,1] (binary) or full matrix (multiclass)
                                               # keras: model.predict(X_test) (sigmoid -> float, softmax -> row)
`test_frame_idx`: the windowing helper (`apply_windowing`) already computes `last_positions` (positions
into the 100 fps base frame). Keep the parallel array of `df.iloc[last_positions, 0]` (the frame column)
and slice it with the same mask used for the test participants. For sequence models the evaluated row is
the last element of each sequence, so take `frame_idx` of that element.
Set `config.track` and `config.label_scheme` in the sweep config (constants per script).

## 3. Sanity check after one run
    python3 - <<'EOF2'
    import pandas as pd, json
    r = pd.read_csv("runs_registry.csv").iloc[-1]
    n = len(json.loads(r.y_pred)); assert n == len(json.loads(r.frame_idx)) == len(json.loads(r.y_proba)), "length mismatch"
    print("ok", n, "rows; fold", r.fold)
    EOF2
