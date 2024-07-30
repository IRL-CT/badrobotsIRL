#exclude keypoints hip and below

import json
import os
import pandas

def json_to_csv(file, flattened_data, frame, participant_num):
    json_data = json.loads(open(file, "r").read())
    data = json_data["people"][0]["pose_keypoints_2d"]
    nose_x = data[0]
    nose_y = data[1]
    nose_c = data[2]
    neck_x = data[3]
    neck_y = data[4]
    neck_c = data[5]
    rightshoulder_x = data[6]
    rightshoulder_y = data[7]
    rightshoulder_c = data[8]
    rightelbow_x = data[9]
    rightelbow_y = data[10]
    rightelbow_c = data[11]
    rightwrist_x = data[12]
    rightwrist_y = data[13]
    rightwrist_c = data[14]
    leftshoulder_x = data[15]
    leftshoulder_y = data[16]
    leftshoulder_c = data[17]
    leftelbow_x = data[18]
    leftelbow_y = data[19]
    leftelbow_c = data[20]
    leftwrist_x = data[21]
    leftwrist_y = data[22]
    leftwrist_c = data[23]
    righteye_x = data[45]
    righteye_y = data[46]
    righteye_c = data[47]
    lefteye_x = data[48]
    lefteye_y = data[49]
    lefteye_c = data[50]
    rightear_x = data[51]
    rightear_y = data[52]
    rightear_c = data[53]
    leftear_x = data[54]
    leftear_y = data[55]
    leftear_c = data[56]
    # background_x = data[75]
    # background_y = data[76]
    # background_c = data[77]
    flattened_data.append({
        "participant" : participant_num,
        "frame" : frame,
        "nose_x" : nose_x,
        "nose_y" : nose_y,
        "nose_c" : nose_c,
        "neck_x" : neck_x,
        "neck_y" : neck_y,
        "neck_c" : neck_c,
        "rightshoulder_x" : rightshoulder_x,
        "rightshoulder_y" : rightshoulder_y,
        "rightshoulder_c" : rightshoulder_c,
        "rightelbow_x" : rightelbow_x,
        "rightelbow_y" : rightelbow_y,
        "rightelbow_c" : rightelbow_c,
        "rightwrist_x" : rightwrist_x,
        "rightwrist_y" : rightwrist_y,
        "rightwrist_c" : rightwrist_c,
        "leftshoulder_x" : leftshoulder_x,
        "leftshoulder_y" : leftshoulder_y,
        "leftshoulder_c" : leftshoulder_c,
        "leftelbow_x" : leftelbow_x,
        "leftelbow_y" : leftelbow_y,
        "leftelbow_c" : leftelbow_c,
        "leftwrist_x" : leftwrist_x,
        "leftwrist_y" : leftwrist_y,
        "leftwrist_c" : leftwrist_c,
        "righteye_x" : righteye_x,
        "righteye_y" : righteye_y,
        "righteye_c" : righteye_c,
        "lefteye_x" : lefteye_x,
        "lefteye_y" : lefteye_y,
        "lefteye_c" : lefteye_c,
        "rightear_x" : rightear_x,
        "rightear_y" : rightear_y,
        "rightear_c" : rightear_c,
        "leftear_x" : leftear_x,
        "leftear_y" : leftear_y,
        "leftear_c" : leftear_c})
    return flattened_data

for i in range(2, 30):

    directory = f"./openpose_nodbot/p{i}nodbot_pose_features"

    if os.path.exists(directory):

        participant_num = directory.split("/")[-1].split("_")[0]
        flattened_data = []

        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            frame = filename.split("_")[1][-4:]
            json_to_csv(file, flattened_data, frame, participant_num)

        features_dataframe = pandas.DataFrame(flattened_data)
        features_dataframe.to_csv(f"./nodbot_features_exclude_lb/{participant_num}_pose_features.csv")