import pandas as pd
file_path = "data/nodbot_annotations.csv"
df = pd.read_csv(file_path)

sorted_data = df.sort_values(by=['participant_num', 'start'])

sorted_file_path = 'data/sorted_nodbot_annotations.csv'
sorted_data.to_csv(sorted_file_path, index=False)
