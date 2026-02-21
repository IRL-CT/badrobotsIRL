"""
SelectKBest feature selection for badrobotsIRL dataset.

Applies scikit-learn's SelectKBest with ANOVA F-value (f_classif) to select
the top-k most discriminative features for binary classification.

Methodology adapted from arXiv:2512.03945 (Schiffmann et al.):
  - Uses SelectKBest with ANOVA F-value analysis
  - Selects the best k features (default k=10)
  - Standardizes features before selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLS = ["frame", "participant", "binary_label", "multiclass_label"]

INPUT_CSV = "interpolation/allparticipants_100fps.csv"
K = 10
TARGET_COL = "binary_label"
OUTPUT_CSV = f"interpolation/select_k_best_allparticipants_100fps_{TARGET_COL}.csv"


def select_k_best_features(df, k=10, target_col="binary_label"):
    """Run SelectKBest (ANOVA F-value) on the feature columns of `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the four info columns and any number of feature columns.
    k : int
        Number of features to select. If k >= total features, all are kept.
    target_col : str
        Column to use as the classification target for the F-test.

    Returns
    -------
    df_selected : pd.DataFrame
        DataFrame with info columns + the top-k features.
    selected_feature_names : list[str]
        Names of the selected features in descending order of score.
    scores : pd.DataFrame
        DataFrame of all features with their F-scores and p-values, sorted
        descending by score.
    """
    info = df[NON_FEATURE_COLS]
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    # Drop columns with zero variance (constant)
    variances = np.var(X, axis=0)
    nonzero_mask = variances > 0
    X_filtered = X[:, nonzero_mask]
    feature_cols_filtered = [feature_cols[i] for i in range(len(feature_cols)) if nonzero_mask[i]]

    # Drop rows with NaN/inf
    finite_mask = np.all(np.isfinite(X_filtered), axis=1)
    X_filtered = X_filtered[finite_mask]
    y_filtered = y[finite_mask]
    info_filtered = info[finite_mask].reset_index(drop=True)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Clamp k to available features
    actual_k = min(k, X_scaled.shape[1])
    print(f"Selecting {actual_k} features out of {X_scaled.shape[1]} "
          f"(requested k={k}, {len(feature_cols) - len(feature_cols_filtered)} "
          f"zero-variance columns dropped)")

    # SelectKBest with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=actual_k)
    selector.fit(X_scaled, y_filtered)

    # Build scores DataFrame
    scores_df = pd.DataFrame({
        "feature": feature_cols_filtered,
        "f_score": selector.scores_,
        "p_value": selector.pvalues_,
    }).sort_values("f_score", ascending=False).reset_index(drop=True)

    # Selected feature names (in descending score order)
    selected_mask = selector.get_support()
    selected_feature_names = [
        feature_cols_filtered[i]
        for i in range(len(feature_cols_filtered))
        if selected_mask[i]
    ]
    selected_feature_names = (
        scores_df[scores_df["feature"].isin(selected_feature_names)]
        ["feature"].tolist()
    )

    # Build output DataFrame with original (non-scaled) values
    df_selected = pd.concat([
        info_filtered,
        pd.DataFrame(
            df[selected_feature_names].values[finite_mask],
            columns=selected_feature_names,
        )
    ], axis=1)

    return df_selected, selected_feature_names, scores_df


def main():
    print(f"Loading dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    print(f"  Shape: {df.shape}")
    print(f"  Participants: {df['participant'].nunique()}")
    print(f"  Feature columns: {len([c for c in df.columns if c not in NON_FEATURE_COLS])}")

    df_selected, selected_features, scores_df = select_k_best_features(
        df, k=K, target_col=TARGET_COL
    )

    print(f"\n=== Top {len(selected_features)} Features (by ANOVA F-score) ===")
    top_scores = scores_df[scores_df["feature"].isin(selected_features)]
    for _, row in top_scores.iterrows():
        print(f"  {row['feature']:>40s}  F={row['f_score']:.4f}  p={row['p_value']:.2e}")

    print(f"\nOutput shape: {df_selected.shape}")
    print(f"{TARGET_COL} distribution:\n{df_selected[TARGET_COL].value_counts().to_string()}")

    df_selected.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")

    scores_df.to_csv(f"select_k_best_all_scores_{TARGET_COL}.csv", index=False)
    print(f"Saved all feature scores to: select_k_best_all_scores_{TARGET_COL}.csv")


if __name__ == "__main__":
    main()
