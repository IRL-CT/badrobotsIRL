import os
import pandas
import csv

directory = "./nodbot_features_labels"

def create_label_csv(directory):

    for filename in os.listdir(directory):

        url = os.path.join(directory, filename)

        df = pandas.read_csv(url)

        df["binary_label"] = 0
        df["multiclass_label"] = 0

        df.to_csv(url, index=False)


def binary_label(url, error_start_frame, error_stop_frame):

    df = pandas.read_csv(url)

    for index, row in df.iterrows():
        frame = float(row[2])
        if frame >= error_start_frame and frame <= error_stop_frame:
            df.at[index, "binary_label"] = 1

    df.to_csv(url, index=False)

def multiclass_label(url, errors):

    df = pandas.read_csv(url)

    for index, row in df.iterrows():
        frame = float(row[2])
        if frame >= errors[0] and frame <= errors[1]: #error 1
            df.at[index, "multiclass_label"] = 1
        elif frame >= errors[1] and frame <= errors[2]: #error 2
            df.at[index, "multiclass_label"] = 2
        elif frame >= errors[2] and frame <= errors[3]: #error 3
            df.at[index, "multiclass_label"] = 3
        elif frame >= errors[3] and frame <= errors[4]: #error 4, for p29 only
            df.at[index, "multiclass_label"] = 4
        elif frame >= errors[4] and frame <= errors[5]: #error 5, for p29 only
            df.at[index, "multiclass_label"] = 5
    
    df.to_csv(url, index=False)


# creating csv with binary and multiclass label columns with default value 0
#create_label_csv(directory)


file = "./nodbot_features_labels/p29nodbot_pose_features_deltas_labels.csv"

binary_label(file, 289, 845)

multiclass_label(file, [289, 519, 682, 873, 1096, 1357])

