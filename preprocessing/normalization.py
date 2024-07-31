import sklearn
from sklearn.preprocessing import StandardScaler
import pandas

file = "preprocessing/merged_features/all_participants.csv"

df = pandas.read_csv(file)

df_norm = df.copy()

scaler = StandardScaler()

features = df_norm.columns[4:]

df_norm[features] = scaler.fit_transform(df_norm[features])

df_norm.to_csv("preprocessing/merged_features/all_participants_normalized.csv", index=False)