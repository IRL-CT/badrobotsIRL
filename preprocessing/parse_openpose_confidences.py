import csv
import os

for i in range(2, 30):

    url = f"./nodbot_features_exclude_lb/p{i}nodbot_pose_features.csv"

    if os.path.exists(url):

        filename = open(url, "r")

        file = csv.DictReader(filename)

        #calculate the occurrence and average confidence of every column/feature

        col_list = {}
        data = {}

        confidence_rows = []

        for row in file:
            for key in row:
                if key.endswith("_c"):
                    if key not in col_list.keys():
                        col_list[key] = []
                        data[key] = [0, 0]
                    if float(row[key]) > 0:
                        col_list[key].append(row[key])

        for key in col_list.keys():
            sum = 0
            length = len(col_list[key])
            for confidences in col_list[key]:
                sum += float(confidences)
            if length != 0:
                average = sum/length
            else:
                average = 0
            confidence_rows.append([key, length, average])
        
        # average of all confidences per participant
        
        all_confidence_sum = 0
        for c in range(len(confidence_rows)):
            all_confidence_sum += confidence_rows[c][2]
        
        average_all_confidence = all_confidence_sum/len(confidence_rows)

        print(f"p{i} average all confidences: {average_all_confidence}")

        # percent of keypoints that have confidence values < 0.5

        low_confidence_vals = []
        total_vals_count = 0.0

        for key in col_list.keys():
            for k in col_list[key]:
                total_vals_count += 1.0
                current_val = float(k)
                if current_val < 0.50:
                    low_confidence_vals.append(current_val)
        
        percent_low_confidence = (len(low_confidence_vals) / total_vals_count) * 100

        print(f"p{i} percent with confidence < 0.50: {percent_low_confidence}")

        # with open(f"./openpose_nodbot/nodbot_exclude_lb_confidences/p{i}nodbot_exclude_lb_confidences.csv", "w", newline="") as file:
        #     writer = csv.writer(file)
        #     field = ["feature", "number_of_identifications", "average_confidence"]
        #     writer.writerow(field)
        #     writer.writerows(confidence_rows)