#SCRIPT to do PCA on Gemini Video Embeddings csv 
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import joblib

def main():
    input_file = '../data/embeddings/gemini_video_embeddings_visual_audio.csv'
    output_file = '../data/embeddings/gemini_video_embeddings_pca_visual_audio.csv'
    model_file = '../data/embeddings/pca_model_gemini_visual_audio.pkl'

    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Make sure you generated the embeddings first.")
        return

    print("Base shape:", df.shape)

    # Columns 0-3 are participant, frame, binary_label, multiclass_label
    df_cols_first = df.iloc[:, 0:4]
    df_feats = df.iloc[:, 4:]

    print("Fitting PCA. To reduce memory and computation time, we use unique rows for fitting...")
    # Since the 1-second interval embeddings are repeated 100 times, 
    # we can just take the unique vectors to fit the PCA. This avoids fitting on 350,000 duplicated rows.
    df_unique_feats = df_feats.drop_duplicates()
    
    print(f"Unique embedding rows for fitting: {df_unique_feats.shape[0]}")
    
    # Do PCA to keep 90% of the variance
    pca = PCA(n_components=0.90)
    pca.fit(df_unique_feats)
    
    print("Total variance retained:", pca.explained_variance_ratio_.sum())
    print("Number of components to reach 90% variance:", pca.n_components_)

    # Transform the FULL 100fps data (all ~350k rows)
    print("Transforming full dataset...")
    df_pca = pca.transform(df_feats)

    # Recombine with metadata 
    full_df = pd.concat([df_cols_first, pd.DataFrame(df_pca)], axis=1)
    
    # Rename the PCA columns
    new_cols = list(df_cols_first.columns) + [f'gemini_PC{i}' for i in range(1, pca.n_components_ + 1)]
    full_df.columns = new_cols
    
    print('Checking for nan values (should be 0):')
    print(full_df.isnull().sum().sum())
    print("Output shape:", full_df.shape)

    print(f"Saving PCA dataset to {output_file}...")
    full_df.to_csv(output_file, index=False)

    print(f"Saving PCA model to {model_file}...")
    joblib.dump(pca, model_file)
    
    print("Done")

if __name__ == '__main__':
    main()
