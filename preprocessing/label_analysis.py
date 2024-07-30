import os
import pandas

directory = "preprocessing/nodbot_features_labels"

def get_binary_details(url):

    df = pandas.read_csv(url)

    total_labels, count_0, count_1 = 0, 0, 0

    for val in df["binary_label"]:
        total_labels += 1
        if val == 0:
            count_0 += 1
        elif val == 1:
            count_1 += 1
    
    return f"binary labels:\n\t0: {count_0}, {round(count_0*100/total_labels, 1)}%\n\t1: {count_1}, {round(count_1*100/total_labels, 1)}%\n\ttotal: {total_labels}"


def get_multiclass_details(url):

    df = pandas.read_csv(url)

    count_0, count_1, count_2, count_3, count_4, count_5, total_labels = 0, 0, 0, 0, 0, 0, 0

    for val in df["multiclass_label"]:
        total_labels += 1
        if val == 0:
            count_0 += 1
        elif val == 1:
            count_1 += 1
        elif val == 2:
            count_2 += 1
        elif val == 3:
            count_3 += 1
        elif val == 4:
            count_4 += 1
        elif val == 5:
            count_5 += 1
    
    if count_4 != 0:
        return f"multiclass labels:\n\t0: {count_0}, {round(count_0*100/total_labels, 1)}%\n\t1: {count_1}, {round(count_1*100/total_labels, 1)}%\n\t2: {count_2}, {round(count_2*100/total_labels, 1)}%\n\t3: {count_3}, {round(count_3*100/total_labels, 1)}%\n\t4: {count_4}, {round(count_4*100/total_labels, 1)}%\n\t5: {count_5}, {round(count_5*100/total_labels, 1)}%\n\ttotal: {total_labels}"
    
    else:
        return f"multiclass labels:\n\t0: {count_0}, {round(count_0*100/total_labels, 1)}%\n\t1: {count_1}, {round(count_1*100/total_labels, 1)}%\n\t2: {count_2}, {round(count_2*100/total_labels, 1)}%\n\t3: {count_3}, {round(count_3*100/total_labels, 1)}%\n\ttotal: {total_labels}"
    

for filename in os.listdir(directory):

    url = os.path.join(directory, filename)

    participant = url.split("\\")[1].split("_")[0]

    print("\n" + participant)
    print(get_binary_details(url))
    print(get_multiclass_details(url))