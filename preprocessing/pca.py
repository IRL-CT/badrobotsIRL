import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("../preprocessing/all_participants_stats_0_3_norm.csv")

participant_frames_labels = df.iloc[:, :4]

x = df.iloc[:, 4:]
x = StandardScaler().fit_transform(x.values)
pca = PCA()
principal_components = pca.fit_transform(x)
print(principal_components.shape)

pca = PCA(n_components=0.90)
principal_components = pca.fit_transform(x)
print(principal_components.shape)

principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

df = principal_df
print(df)
df.to_csv("../preprocessing/all_participants_stats_0_3_norm_pca.csv", index=False)
