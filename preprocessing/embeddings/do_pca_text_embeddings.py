# SCRIPT to do PCA on text embeddings csv 
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

def main():
    # Load text embeddings
    text_path = os.path.join(DATA_DIR, "clip_text_embeddings.csv")
    if not os.path.exists(text_path):
        text_path = os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings.csv")

    df = pd.read_csv(text_path)
    print(df.head())

    df_cols_first = df.iloc[:, 0:4]
    df_feats = df.iloc[:, 4:]
    print('DF COLS FIRST', df_cols_first.head())

    # Do PCA to keep 90% of the variance
    pca = PCA(n_components=0.90)
    pca.fit(df_feats)
    print("Explained variance ratio sum:", pca.explained_variance_ratio_.sum())
    print("Number of components:", pca.n_components_)

    # Transform data
    df_pca = pca.transform(df_feats)

    # Save PCA data as CSV
    full_df = pd.concat([df_cols_first, pd.DataFrame(df_pca)], axis=1, ignore_index=True)
    new_cols = ['participant', 'frame', 'binary_label', 'multiclass_label'] + [f'PC{i}' for i in range(1, df_pca.shape[1] + 1)]
    full_df.columns = new_cols
    full_df = full_df.reset_index(drop=True)

    out_pca_path = os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings_pca.csv")
    full_df.to_csv(out_pca_path, index=False)
    # Also save to data/ for backward compatibility if needed
    full_df.to_csv(os.path.join(DATA_DIR, "clip_text_embeddings_pca.csv"), index=False)

    # Save PCA model
    joblib.dump(pca, os.path.join(EMBEDDINGS_DIR, "pca_model_text.pkl"))

    # Calculate cosine distance between consecutive rows
    df_orig = pd.read_csv(text_path)
    df_orig_cols_first = df_orig.iloc[:, 0:4]
    df_orig_feats = df_orig.iloc[:, 4:]

    similarity_list = []
    for i in range(len(df_orig_feats) - 1):
        row1 = df_orig_feats.iloc[i].values.reshape(1, -1)
        row2 = df_orig_feats.iloc[i+1].values.reshape(1, -1)
        similarity = cosine_similarity(row1, row2)[0][0]
        similarity_list.append(similarity)

    print('similarity list length', len(similarity_list))
    cosine_df = df_orig_cols_first.copy()
    cosine_df.columns = ['participant', 'frame', 'binary_label', 'multiclass_label']
    cosine_df['Distance'] = [np.nan] + similarity_list
    print(cosine_df.head())

    out_cos_path = os.path.join(EMBEDDINGS_DIR, "clip_text_cosine_similarity.csv")
    cosine_df.to_csv(out_cos_path, index=False)
    cosine_df.to_csv(os.path.join(DATA_DIR, "clip_text_cosine_similarity.csv"), index=False)
    print("Done! Text PCA and cosine similarity saved.")

if __name__ == '__main__':
    main()
