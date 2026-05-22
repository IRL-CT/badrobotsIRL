import pandas as pd

rnn_df = pd.read_csv("rnn_transformer_models_5_21.csv")
linear_df = pd.read_csv("linear_models.csv")

rnn_df = rnn_df.rename(columns={"Name": "run_name"})

linear_df = linear_df.rename(columns={
    "Name": "run_name",
    "model_type": "model",
    "binary_multiclass": "class",
})

rnn_df = rnn_df[rnn_df["State"] == "finished"]
linear_df = linear_df[linear_df["State"] == "finished"]

print(f"RNN/Transformer finished runs: {len(rnn_df)}")
print(f"Linear finished runs: {len(linear_df)}")

rnn_models = ["gru", "lstm"]
linear_models = ["svm", "rf", "sgd", "mlp", "knn"]

rnn_df = rnn_df[rnn_df["model"].isin(rnn_models)].copy()
linear_df = linear_df[linear_df["model"].isin(linear_models)].copy()

print(f"RNN runs after model filter: {len(rnn_df)}")
print(f"Linear runs after model filter: {len(linear_df)}")

# Calculate standard deviations from fold metrics
base_metrics = [
    "accuracy", "f1", "precision", "recall", "auc", 
    "windowed_accuracy", "windowed_f1", "windowed_precision", "windowed_recall", "fnr"
]

for m in base_metrics:
    # RNN fold metrics: fold_0_metrics.test_f1, etc.
    rnn_cols = [f"fold_{i}_metrics.test_{m}" for i in range(5)]
    rnn_exist = [c for c in rnn_cols if c in rnn_df.columns]
    if rnn_exist:
        rnn_df[f"std_test_{m}"] = rnn_df[rnn_exist].std(axis=1)

    # Linear fold metrics: t0_test_f1, etc.
    lin_cols = [f"t{i}_test_{m}" for i in range(5)]
    lin_exist = [c for c in lin_cols if c in linear_df.columns]
    if lin_exist:
        linear_df[f"std_test_{m}"] = linear_df[lin_exist].std(axis=1)

# performance metrics
perf_cols = []
for m in base_metrics:
    perf_cols.extend([f"avg_test_{m}", f"std_test_{m}"])

id_cols = ["run_name", "class", "model", "modality", "feature_set", "Sweep", "Created"]

# RNN hyperparameters
rnn_hyper_cols = [
    "activation_function", "aggregation", "batch_size", "dense_units",
    "dropout_rate", "epochs", "fusion_type", "learning_rate", "loss",
    "lstm_units", "gru_units", "num_lstm_layers", "num_gru_layers",
    "optimizer", "recurrent_regularizer", "sequence_length", "stride",
    "use_bidirectional", "window", "agg_features", "gemini_dims",
    "test_stride", "test_window_size",
]

# Linear hyperparameters
linear_hyper_cols = [
    "aggregation", "C", "alpha", "gamma", "kernel", "max_depth", "max_iter",
    "mlp_activation", "mlp_hidden_layer_sizes", "n_estimators", "n_neighbors",
    "tol", "stride", "window", "agg_features", "gemini_embedding",
    "test_stride", "test_window_size",
]

def pick_cols(df, col_list):
    """Return only columns that actually exist in the dataframe."""
    return [c for c in col_list if c in df.columns]

rnn_keep = pick_cols(rnn_df, id_cols + perf_cols + rnn_hyper_cols)
rnn_clean = rnn_df[rnn_keep].copy()

linear_keep = pick_cols(linear_df, id_cols + perf_cols + linear_hyper_cols)
linear_clean = linear_df[linear_keep].copy()

df = pd.concat([rnn_clean, linear_clean], ignore_index=True)

df = df.dropna(subset=["avg_test_f1"]) # drop runs with no f1
print(f"\nTotal runs with valid F1: {len(df)}")

# Tag each run as rnn or linear
dl_models = ["gru", "lstm"]
df["model_type"] = df["model"].apply(lambda m: "rnn" if m in dl_models else "linear")

# Fix modality column to reflect actual data used
NO_MODALITY_SELECTION_SETS = {"catch22", "tsfresh", "curated_features_v5_100fps", "rf", "selectkbest", "curated_v4", "curated_v5"}

def get_effective_modality(row):
    feat_set = row.get("feature_set", "")
    mod = str(row.get("modality", ""))
    is_rnn = row.get("model_type") == "rnn"
    
    if feat_set in NO_MODALITY_SELECTION_SETS:
        if is_rnn:
            components = mod.split('_')
            text_mods = [c for c in components if c in ["text", "cosine", "gemini"]]
            if text_mods:
                return feat_set + "_" + "_".join(text_mods)
            else:
                return feat_set
        else:
            # Linear models do not append text/cosine/gemini in NO_MODALITY_SELECTION_SETS
            return feat_set
    return mod

df["modality"] = df.apply(get_effective_modality, axis=1)


df = df.sort_values("avg_test_f1", ascending=False)

# Best run per individual model per (class, model, modality) — for the CSV
best_all = df.groupby(["class", "model", "modality"], dropna=False).first().reset_index()
best_all = best_all.sort_values(["class", "modality", "model_type", "model"])

print(f"Top performing runs: {len(best_all)}\n")
print(best_all[["class", "model_type", "model", "modality", "feature_set", "avg_test_f1", "Sweep"]].to_string(index=False))

# Best rnn and linear per (class, modality) for comparison
df_dl = df[df["model_type"] == "rnn"]
best_dl = df_dl.groupby(["class", "modality"], dropna=False).first().reset_index()
df_lin = df[df["model_type"] == "linear"]
best_lin = df_lin.groupby(["class", "modality"], dropna=False).first().reset_index()

summary_rows = []
for cls in ["binary", "multiclass"]:
    dl_sub = best_dl[best_dl["class"] == cls].set_index("modality")
    lin_sub = best_lin[best_lin["class"] == cls].set_index("modality")
    shared = sorted(set(dl_sub.index) & set(lin_sub.index))
    for mod in shared:
        dl_f1 = dl_sub.loc[mod, "avg_test_f1"]
        dl_model = dl_sub.loc[mod, "model"]
        lin_f1 = lin_sub.loc[mod, "avg_test_f1"]
        lin_model = lin_sub.loc[mod, "model"]
        if isinstance(dl_f1, pd.Series):
            dl_f1, dl_model = dl_f1.iloc[0], dl_model.iloc[0]
        if isinstance(lin_f1, pd.Series):
            lin_f1, lin_model = lin_f1.iloc[0], lin_model.iloc[0]
        diff = lin_f1 - dl_f1
        summary_rows.append({
            "class": cls, "modality": mod,
            "best_rnn": dl_model, "rnn_f1": round(dl_f1, 4),
            "best_linear": lin_model, "linear_f1": round(lin_f1, 4),
            "winner": "linear" if diff > 0 else "rnn",
            "f1_diff": round(abs(diff), 4),
        })

summary_df = pd.DataFrame(summary_rows)

# Write both to a single CSV with a separator
with open("top_models_clarity.csv", "w") as f:
    best_all.to_csv(f, index=False)
    f.write("\n")
    f.write("RNN vs Linear Summary\n")
    summary_df.to_csv(f, index=False)

print("\nSaved to top_models_clarity.csv")