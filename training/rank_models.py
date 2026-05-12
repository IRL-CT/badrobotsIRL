import pandas as pd

rnn_df = pd.read_csv("rnn_transformer_models.csv")
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

# performance metrics
perf_cols = [
    "avg_test_accuracy", "avg_test_f1", "avg_test_precision", "avg_test_recall",
    "avg_test_auc", "avg_test_windowed_accuracy", "avg_test_windowed_f1",
    "avg_test_windowed_precision", "avg_test_windowed_recall", "avg_test_fnr",
]

id_cols = ["run_name", "class", "model", "modality", "feature_set", "Sweep"]

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

# Select top performing run per (class, model, modality) by f1 score

df = df.sort_values("avg_test_f1", ascending=False)
best = df.groupby(["class", "model", "modality"], dropna=False).first().reset_index()
best = best.sort_values(["class", "modality", "model"])

print(f"Top performing runs: {len(best)}\n")
print(best[["class", "model", "modality", "feature_set", "avg_test_f1", "Sweep"]].to_string(index=False))

best.to_csv("top_models.csv", index=False)
print("\nSaved to top_models.csv")