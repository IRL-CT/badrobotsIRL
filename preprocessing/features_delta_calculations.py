import os
import pandas
import csv

for i in range(2, 30):

    url = f"./nodbot_features_exclude_lb/p{i}nodbot_pose_features.csv"

    if os.path.exists(url):

        df = pandas.read_csv(url)

        features = []

        for col in df.columns:
            if col.endswith("_c"):
                df = df.drop([col], axis=1)
            else:
                if col != "Unnamed: 0" and col != "participant" and col != "frame":
                    features.append(col)
                    position = df.columns.get_loc(col) + 1
                    df.insert(position, f"{col}_delta", 0.0)
        
        total_frames = len(df)

        for feature in features:

            for frame in range(total_frames-1):

                val_0 = df.loc[frame, f"{feature}"]
                val_1 = df.loc[frame+1, f"{feature}"]
                delta = float(val_1) - float(val_0)

                df.loc[frame+1, f"{feature}_delta"] = round(delta, 4)

        df.rename(columns={"Unnamed: 0" : "row"})        

        df.to_csv(f"./nodbot_features_deltas/p{i}nodbot_pose_features_deltas.csv", index=False)