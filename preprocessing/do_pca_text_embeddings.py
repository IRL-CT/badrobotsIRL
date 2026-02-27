#SCRIPT to do PCA on text embeddings csv 
#import
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA



def main():
        
    #load the csv file
    df = pd.read_csv('../data/clip_text_embeddings.csv')
    print(df.head())

    #select only columsn 2:end
    
    df_cols_first = df.iloc[:,0:2]
    df_feats = df.iloc[:,2:]
    print('DF COLS FIRST', df_cols_first.head())

    #do PCA to keep 95% of the variance
    pca = PCA(n_components=0.90)
    pca.fit(df_feats)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    #see how many components we have
    print(pca.n_components_)

    #transform the data
    df_pca = pca.transform(df_feats)

    #save the pca data as csv
    full_df = pd.concat([df_cols_first, pd.DataFrame(df_pca)], axis=1, ignore_index=True)
    #rename the columns
    new_cols = ['participant', 'frame'] + [f'PC{i}' for i in range(1, df_pca.shape[1] + 1)]
    full_df.columns = new_cols
    #check for nan
    print('checking for nan values')
    print(full_df.isnull().sum())
    print(full_df.shape)
    #remove rows with nan values
    #full_df = full_df.dropna()
    #new index
    full_df = full_df.reset_index(drop=True)
    full_df.to_csv('clip_text_embeddings_pca.csv', index=False)

    #save the pca model
    import joblib
    joblib.dump(pca, 'pca_model.pkl')

    #now, back in the original df, get the cosine distance between every two rows, start with a zero row so it's the same shape
    #reload original df for cosine similarity calculation
    df_orig = pd.read_csv('../data/clip_text_embeddings.csv')
    df_orig_cols_first = df_orig.iloc[:,0:2]
    df_orig_feats = df_orig.iloc[:,2:]
    
    #now, calculate cosine distance
    from sklearn.metrics.pairwise import cosine_similarity
    # Calculate cosine similarity for consecutive rows
    similarity_list = []
    for i in range(len(df_orig_feats) - 1):
        row1 = df_orig_feats.iloc[i].values.reshape(1, -1)
        row2 = df_orig_feats.iloc[i+1].values.reshape(1, -1)
        similarity = cosine_similarity(row1, row2)[0][0]
        similarity_list.append(similarity)

    # print similatiry list length
    print('similarity list length', len(similarity_list))
    # Add similarity values to DataFrame
    cosine_df = df_orig_cols_first.copy()
    cosine_df.columns = ['participant', 'frame']
    cosine_df['Distance'] = [np.nan] + similarity_list
    print(cosine_df.head())
    print(cosine_df.shape)
    

    #save cosine df and df
    cosine_df.to_csv('../data/clip_text_cosine_similarity.csv', index=False)
    #save the cosine similarity model

if __name__ == '__main__':
    main()