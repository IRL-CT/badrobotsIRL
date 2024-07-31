import pandas
import numpy as np

file = "preprocessing/merged_features/all_participants.csv"

df = pandas.read_csv(file)

nan_count = df.isna().sum().sum()

print(f"Number of NaNs: {nan_count}")

numeric_cols = df.select_dtypes(include=[np.number])
inf_count = np.isinf(numeric_cols).sum().sum()

print(f"Number of infs: {inf_count}")

df.fillna(0, inplace=True)

df.replace([np.inf, -np.inf], 0, inplace=True)

print(df)

# only needs to be executed if number of NaN or number of infs > 0
# df.to_csv("preprocessing/merged_features/all_participants_final.csv", index=False)