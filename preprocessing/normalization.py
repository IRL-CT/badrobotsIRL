import sklearn
from sklearn.preprocessing import StandardScaler
import pandas

file = "../preprocessing/all_participants_rf_0_3.csv"

df = pandas.read_csv(file)

df_norm = df.copy()

scaler = StandardScaler()

features = df_norm.columns[4:]

df_norm[features] = scaler.fit_transform(df_norm[features])

df_norm.to_csv("../preprocessing/all_participants_rf_0_3_norm.csv", index=False)

print(df_norm)