#!/usr/bin/env python3
"""Hierarchical escalation model (v6): window encoder + session encoder.

The hypothesis
--------------
Cross-user performance plateaus (~0.69 AUC) well below within-user (~0.83). One
reason is that every window is currently classified in isolation: the model sees
one chunk of behaviour and must decide "error or not" on absolute appearance.
Person A's agitated may look like Person B's neutral, so absolute appearance does
not transfer.

But the experiment has structure that is being thrown away: each participant gets
THREE errors in a row and reactions escalate, and each starts from a quiet survey
baseline. "More agitated than this same person was a minute ago" is a RELATIVE
quantity, and relative escalation should transfer across people much better than
absolute expression does.

    level 1   RNN over the seconds inside one window        -> window embedding
    level 2   RNN over the windows of the session, in order -> per-window state

Level 2 carries hidden state from the survey through Error I -> II -> III, so the
head sees the person's own trajectory rather than an isolated snapshot. It steps
over fixed time windows, never over annotated blocks, so nothing here needs
boundary knowledge at test time.

Controls that matter
--------------------
`level2: "none"` collapses this to a flat per-window model. That is the control
that says whether the hierarchy earns its keep -- without it a gain could just be
the 1 Hz resampling or the v4 labels. `level1: "mean"` likewise ablates the
window encoder.

Honest expectations
-------------------
24 participants x ~40 s is a very small number of sequences for a two-level
recurrent model. Heavy regularisation is on by default and it may simply not have
the data. The explicit per-person baselining version of this idea was already
tested and made things slightly WORSE (0.638 -> 0.615 AUC), so the learned
version is a different bet, not a safe one. Treat `level2: "none"` beating the
recurrent variants as a real possible outcome.

The baseline is not 0.5. With the embeddings replaced by pure Gaussian noise and
the session structure kept, the sibling escalation-feature model still scores
0.55-0.61 AUC, because the first windows of a session have structurally smaller
deltas and the first windows are always the survey (label 0). Judge these runs
against that floor, not against chance. See the note in escalation_linear.py.
"""

import argparse
import os
import sys

import numpy as np
import wandb
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "common"))

import hier_data as H  # noqa: E402
from eval_utils import _core_metrics, participant_bootstrap  # noqa: E402

import tensorflow as tf  # noqa: E402
from keras import layers, models, optimizers, regularizers  # noqa: E402


def _rnn(kind, units, return_sequences, reg, dropout):
    common = dict(return_sequences=return_sequences, dropout=dropout,
                  recurrent_regularizer=reg)
    if kind in ("gru", "bigru"):
        cell = layers.GRU(units, **common)
    elif kind in ("lstm", "bilstm"):
        cell = layers.LSTM(units, **common)
    elif kind in ("rnn", "birnn"):
        cell = layers.SimpleRNN(units, **common)
    else:
        raise ValueError(f"unknown rnn {kind!r}")
    return layers.Bidirectional(cell) if kind.startswith("bi") else cell


def build_model(cfg, n_windows, window_sec, n_features, n_classes):
    reg = regularizers.l2(cfg.l2_reg) if cfg.l2_reg else None
    inp = layers.Input(shape=(n_windows, window_sec, n_features))

    # ---- level 1: encode the seconds inside each window ---------------------
    if cfg.level1 == "mean":
        h = layers.TimeDistributed(layers.GlobalAveragePooling1D())(inp)
    else:
        h = layers.TimeDistributed(
            _rnn(cfg.level1, cfg.units_l1, False, reg, cfg.dropout))(inp)
    h = layers.TimeDistributed(layers.LayerNormalization())(h)

    # ---- level 2: carry state across the session ----------------------------
    # Deliberately causal: a deployed robot cannot see the future of the session.
    if cfg.level2 != "none":
        h = _rnn(cfg.level2, cfg.units_l2, True, reg, cfg.dropout)(h)
        h = layers.TimeDistributed(layers.LayerNormalization())(h)

    h = layers.TimeDistributed(layers.Dropout(cfg.dropout))(h)
    out = layers.TimeDistributed(
        layers.Dense(n_classes, activation="softmax"), name="cls")(h)

    outputs = [out]
    if cfg.aux_localization:
        # REPAIR-Bench pairs classification with a Huber regression on WHEN the
        # failure happened. With five contiguous blocks per session, onset
        # regression is better posed than thousands of independent decisions.
        outputs.append(layers.TimeDistributed(
            layers.Dense(1, activation="linear"), name="onset")(h))

    model = models.Model(inp, outputs if len(outputs) > 1 else out)
    opt = {"adam": optimizers.Adam, "rmsprop": optimizers.RMSprop,
           "sgd": optimizers.SGD}[cfg.optimizer](learning_rate=cfg.learning_rate)

    if cfg.aux_localization:
        model.compile(
            optimizer=opt,
            loss={"cls": tf.keras.losses.CategoricalCrossentropy(
                      label_smoothing=cfg.label_smoothing),
                  "onset": tf.keras.losses.Huber(delta=4.0)},
            loss_weights={"cls": 1.0, "onset": cfg.aux_weight},
            weighted_metrics=[],   # silences the sample_weight/metrics warning
            sample_weight_mode="temporal")
    else:
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=cfg.label_smoothing),
            weighted_metrics=[],   # silences the sample_weight/metrics warning
            sample_weight_mode="temporal")
    return model


def _onset_target(sessions, n_windows):
    """Per window: signed distance in windows to the first error, /10.

    Auxiliary target only; scaled so Huber(delta=4) sees a sane range.
    """
    T = np.zeros((len(sessions), n_windows, 1), dtype="float32")
    for i, s in enumerate(sessions):
        err = np.flatnonzero(s["y"] > 0)
        first = err[0] if len(err) else len(s["y"])
        T[i, :len(s["y"]), 0] = (np.arange(len(s["y"])) - first) / 10.0
    return T


def train():
    wandb.init()
    cfg = wandb.config

    info, X = H.load_frames(cfg.modality, cfg.gemini_source, cfg.gemini_dims,
                            feature_set=getattr(cfg, "feature_set", "full"),
                            binary_multiclass=cfg.binary_multiclass)
    Xs, yb, ym, parts = H.to_seconds(info, X)
    y = yb if cfg.binary_multiclass == "binary" else ym
    n_classes = int(y.max()) + 1
    wandb.log({"n_seconds": len(Xs), "n_features": Xs.shape[1],
               "n_classes": n_classes})

    folds = H.participant_folds(parts)
    fold_metrics, all_pred, all_true, all_part, all_proba = [], [], [], [], []

    for fold, (tr_p, va_p, te_p) in enumerate(folds):
        # Scale on TRAIN participants only -- the scaler is part of the model.
        tr_mask = np.isin(parts, tr_p)
        scaler = StandardScaler().fit(Xs[tr_mask])
        Xz = scaler.transform(Xs).astype("float32")

        sess = H.build_sessions(Xz, y, parts, cfg.window_sec, cfg.stride_sec)
        by_part = {s["part"]: s for s in sess}
        split = {k: [by_part[p] for p in ps if p in by_part]
                 for k, ps in (("tr", tr_p), ("va", va_p), ("te", te_p))}
        if not all(split.values()):
            print(f"fold {fold}: empty split, skipping")
            continue

        n_windows = max(len(s["y"]) for s in sess)
        packed = {}
        for k, ss in split.items():
            Xp = np.zeros((len(ss), n_windows, cfg.window_sec, Xz.shape[1]), "float32")
            Yp = np.zeros((len(ss), n_windows, n_classes), "float32")
            SW = np.zeros((len(ss), n_windows), "float32")
            for i, s in enumerate(ss):
                w = len(s["y"])
                Xp[i, :w] = s["X"]
                Yp[i, np.arange(w), s["y"].astype(int)] = 1.0
                SW[i, :w] = 1.0
            packed[k] = (Xp, Yp, SW, ss)

        # Inverse-frequency class weights folded into the sample weights, so
        # padding stays at 0 and rare classes still count.
        ytr = np.concatenate([s["y"] for s in split["tr"]])
        counts = np.bincount(ytr.astype(int), minlength=n_classes).astype("float64")
        cw = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_classes, 0.0)
        for k in packed:
            Xp, Yp, SW, ss = packed[k]
            lab = Yp.argmax(-1)
            packed[k] = (Xp, Yp, SW * cw[lab], ss)

        model = build_model(cfg, n_windows, cfg.window_sec, Xz.shape[1], n_classes)

        def pack(k):
            Xp, Yp, SW, ss = packed[k]
            if cfg.aux_localization:
                return Xp, {"cls": Yp, "onset": _onset_target(ss, n_windows)}, \
                       {"cls": SW, "onset": (SW > 0).astype("float32")}
            return Xp, Yp, SW

        xtr, ytr_d, swtr = pack("tr")
        xva, yva_d, swva = pack("va")
        model.fit(xtr, ytr_d, sample_weight=swtr,
                  validation_data=(xva, yva_d, swva),
                  epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=cfg.patience,
                      restore_best_weights=True)])

        Xte, Yte, SWte, ste = packed["te"]
        pr = model.predict(Xte, verbose=0)
        if cfg.aux_localization:
            pr = pr[0]
        keep = SWte > 0
        proba = pr[keep]
        true = Yte.argmax(-1)[keep]
        pred = proba.argmax(-1)
        part_of = np.concatenate([np.full(len(s["y"]), s["part"]) for s in ste])

        m = _core_metrics(pred, true, proba)
        fold_metrics.append(m)
        # Same key shape as v4/v5 so all sweeps compare in one W&B view.
        wandb.log({f"t{fold}_test_{k}": v for k, v in m.items()})
        print(f"fold {fold}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        all_pred.append(pred); all_true.append(true); all_part.append(part_of)
        all_proba.append(proba)   # needed for pooled AUC; without it it is NaN

        tf.keras.backend.clear_session()

    if not fold_metrics:
        wandb.log({"status": "no_valid_folds"})
        return

    avg = {f"avg_test_{k}": float(np.mean([m[k] for m in fold_metrics]))
           for k in fold_metrics[0]}
    wandb.log(avg)

    # Pooled participant bootstrap: the honest interval at 24 participants.
    P = np.concatenate(all_pred); T = np.concatenate(all_true)
    G = np.concatenate(all_part); PR = np.vstack(all_proba)
    # The evaluation unit is already the window, so the pooled estimate and its
    # participant bootstrap take the sec_test_* names v4/v5 use.
    # PR (not None) is essential: _core_metrics returns AUC = NaN without
    # probabilities, which is what made avg_sec_test_auc NaN on the first runs.
    wandb.log({f"avg_sec_test_{k}": v for k, v in _core_metrics(P, T, PR).items()})
    wandb.log({f"avg_sec_test_{k}": v for k, v in
               participant_bootstrap(P, T, G, PR, n_bootstrap=500).items()})
    wandb.log({"avg_n_eval_units": int(len(T)),
               "avg_n_eval_participants": int(len(set(G)))})
    print("avg:", avg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="brirl_hier_escalation_v6")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--count", type=int, default=None)
    args = ap.parse_args()

    sweep = {
        "method": "random",
        "name": "brirl_hier_escalation_v6",
        "parameters": {
            "binary_multiclass": {"values": ["binary", "multiclass"]},
            # --- the hierarchy itself ---------------------------------------
            "level1": {"values": ["gru", "lstm", "rnn", "bigru", "bilstm", "mean"]},
            # "none" is the flat control: no session-level state at all.
            "level2": {"values": ["gru", "lstm", "rnn", "none"]},
            "units_l1": {"values": [32, 64, 128]},
            "units_l2": {"values": [32, 64, 128]},
            # --- temporal granularity (in SECONDS, data is resampled to 1 Hz) -
            # Sessions are only ~29 s after the recovery block is dropped (min 23,
            # max 43), so a long window is most of the interaction. Fraction of
            # windows that contain a SINGLE label:
            #     window_sec   binary   multiclass(4-class)
            #          1        100%        100%
            #          2         97%         90%
            #          3         93%         79%
            #          5         85%         54%
            #         10         58%         10%   <- removed
            # Error I averages only ~5.5 s per participant, so a 10 s window
            # cannot sit inside it at all: it gets its last second's label while
            # mostly containing something else. That is mislabelling, not
            # smoothing. OLD: [3, 5, 10].
            "window_sec": {"values": [1, 2, 3, 5]},
            # Stride is clamped to the window at build time; 5 was only useful
            # for window_sec=10. OLD: [1, 2, 5].
            "stride_sec": {"values": [1, 2]},
            # --- regularisation: 24 sessions is very little data -------------
            "dropout": {"values": [0.2, 0.4, 0.6]},
            "l2_reg": {"values": [0.0, 1e-4, 1e-3]},
            "label_smoothing": {"values": [0.0, 0.05]},
            # --- optimisation -------------------------------------------------
            "optimizer": {"values": ["adam", "rmsprop"]},
            "learning_rate": {"values": [1e-3, 5e-4, 1e-4]},
            "batch_size": {"values": [4, 8]},
            "epochs": {"values": [100]},
            "patience": {"values": [15]},
            # --- auxiliary onset regression (REPAIR-Bench Task 1) -------------
            "aux_localization": {"values": [True, False]},
            "aux_weight": {"values": [0.5, 2.0]},
            # --- features -----------------------------------------------------
            # Conventional-feature source. Curated sets are one undivided block;
            # they still compose with text/cosine/gemini (see feature_sets.py).
            "feature_set": {"values": ["full", "curated_v5", "rf", "selectkbest"]},
            "gemini_source": {"values": ["visual", "visual_audio"]},
            "gemini_dims": {"values": [128, 256, 768]},
            "modality": {"values": [
                "gemini", "facial", "audio", "pose",
                "facial_audio", "pose_facial_audio",
                "facial_gemini", "audio_gemini", "pose_gemini",
                "facial_audio_gemini", "pose_facial_audio_gemini",
            ]},
        },
    }
    print(sweep)
    sid = wandb.sweep(sweep=sweep, project=args.project, entity=args.entity)
    wandb.agent(sid, function=train, count=args.count,
                entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
