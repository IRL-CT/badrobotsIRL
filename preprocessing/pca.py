import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("../preprocessing/full_features/all_participants_0_3.csv")
df_text = pd.read_csv("../preprocessing/text_embeddings.csv")

participant_frames_labels = df.iloc[:, :4]
participant_frames = df_text.iloc[:, :2]

x = df.iloc[:, 2:]
x = StandardScaler().fit_transform(x.values)
pca = PCA()
principal_components = pca.fit_transform(x)
print(principal_components.shape)

pca = PCA(n_components=0.90)
principal_components = pca.fit_transform(x)
print(principal_components.shape)

principal_df = pd.DataFrame(data=principal_components, columns=['principal component ' + str(i) for i in range(principal_components.shape[1])])
principal_df = pd.concat([participant_frames_labels, principal_df], axis=1)

print(participant_frames_labels)
print(principal_df)
#principal_df.to_csv("../preprocessing/text_embeddings_pca.csv", index=False)
