import json
import os
import pandas
import csv

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
    midhip_x = data[24]
    midhip_y = data[25]
    midhip_c = data[26]
    righthip_x = data[27]
    righthip_y = data[28]
    righthip_c = data[29]
    rightknee_x = data[30]
    rightknee_y = data[31]
    rightknee_c = data[32]
    rightankle_x = data[33]
    rightankle_y = data[34]
    rightankle_c = data[35]
    lefthip_x = data[36]
    lefthip_y = data[37]
    lefthip_c = data[38]
    leftknee_x = data[39]
    leftknee_y = data[40]
    leftknee_c = data[41]
    leftankle_x = data[42]
    leftankle_y = data[43]
    leftankle_c = data[44]
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
    leftbigtoe_x = data[57]
    leftbigtoe_y = data[58]
    leftbigtoe_c = data[59]
    leftsmalltoe_x = data[60]
    leftsmalltoe_y = data[61]
    leftsmalltoe_c = data[62]
    leftheel_x = data[63]
    leftheel_y = data[64]
    leftheel_c = data[65]
    rightbigtoe_x = data[66]
    rightbigtoe_y = data[67]
    rightbigtoe_c = data[68]
    rightsmalltoe_x = data[69]
    rightsmalltoe_y = data[70]
    rightsmalltoe_c = data[71]
    rightheel_x = data[72]
    rightheel_y = data[73]
    rightheel_c = data[74]
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
        "midhip_x" : midhip_x,
        "midhip_y" : midhip_y,
        "midhip_c" : midhip_c,
        "righthip_x" : righthip_x,
        "righthip_y" : righthip_y,
        "righthip_c" : righthip_c,
        "rightknee_x" : rightknee_x,
        "rightknee_y" : rightknee_y,
        "rightknee_c" : rightknee_c,
        "rightankle_x" : rightankle_x,
        "rightankle_y" : rightankle_y,
        "rightankle_c" : rightankle_c,
        "lefthip_x" : lefthip_x,
        "lefthip_y" : lefthip_y,
        "lefthip_c" : lefthip_c,
        "leftknee_x" : leftknee_x,
        "leftknee_y" : leftknee_y,
        "leftknee_c" : leftknee_c,
        "leftankle_x" : leftankle_x,
        "leftankle_y" : leftankle_y,
        "leftankle_c" : leftankle_c,
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
        "leftear_c" : leftear_c,
        "leftbigtoe_x" : leftbigtoe_x,
        "leftbigtoe_y" : leftbigtoe_y,
        "leftbigtoe_c" : leftbigtoe_c,
        "leftsmalltoe_x" : leftsmalltoe_x,
        "leftsmalltoe_y" : leftsmalltoe_y,
        "leftsmalltoe_c" : leftsmalltoe_c,
        "leftheel_x" : leftheel_x,
        "leftheel_y" : leftheel_y,
        "leftheel_c" : leftheel_c,
        "rightbigtoe_x" : rightbigtoe_x,
        "rightbigtoe_y" : rightbigtoe_y,
        "rightbigtoe_c" : rightbigtoe_c,
        "rightsmalltoe_x" : rightsmalltoe_x,
        "rightsmalltoe_y" : rightsmalltoe_y,
        "rightsmalltoe_c" : rightsmalltoe_c,
        "rightheel_x" : rightheel_x,
        "rightheel_y" : rightheel_y,
        "rightheel_c" : rightheel_c})
    return flattened_data


for i in range(2, 30):

    directory = "./p29nodbot_pose_features"

    if os.path.exists(directory):

        participant_num = directory.split("/")[-1].split("_")[0]
        flattened_data = []

        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            frame = filename.split("_")[1][-4:]
            json_to_csv(file, flattened_data, frame, participant_num)

        features_dataframe = pandas.DataFrame(flattened_data)

        features_dataframe.to_csv(f"./nodbot_csv_raw/{participant_num}_pose_features.csv")
