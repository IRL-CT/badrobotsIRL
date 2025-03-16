#SCRIPT to do PCA on text embeddings csv 
#import
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA



def main():
        
    #load the csv file
    df = pd.read_csv('./text_embeddings.csv')
    df_orig = pd.read_csv("../preprocessing/full_features/all_participants_0_3.csv")
    participant_frames_labels = df_orig.iloc[:, :4]

    #select only columsn 2:end
    df_cols_first = df.iloc[:,:2]
    df = df.iloc[:,2:]

    #do PCA to keep 95% of the variance
    pca = PCA(n_components=0.95)
    pca.fit(df)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    #see how many components we have
    print(pca.n_components_)

    #transform the data
    df_pca = pca.transform(df)

    #save the pca data as csv
    full_df = pd.concat([participant_frames_labels, pd.DataFrame(df_pca)], axis=1)
    print(full_df)
    full_df.to_csv('text_embeddings_pca.csv', index=False)

    #save the pca model
    import joblib
    joblib.dump(pca, 'pca_model.pkl')


if __name__ == '__main__':
    main()