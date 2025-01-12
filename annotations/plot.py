import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

annotation_list = [
     #"prompt", "repeated prompt", "more specific/longer prompt", "swapping terms in prompt", "simpler prompt",
     #"slower", "demanding tone", "interrogative tone", "filler words",
     "amusement", "confusion", "frustration", "humor"
    #  "quitting", "looking at tablet",
    # "looking at door",
]

annotation_name = "Emotion"

data = pd.read_csv('sorted_nodbot_annotations_revised.csv')

data['labels'] = data['labels'].str.split(", ").apply(lambda x: [label for label in x if label in annotation_list])

data = data.explode('labels')

grouped_data = data.groupby(['error', 'labels']).size().unstack(fill_value=0)

grouped_data = grouped_data.reindex(columns=annotation_list, fill_value=0)

errors = grouped_data.index
annotations = grouped_data.columns
num_annotations = len(annotations)
bar_width = 0.8 / num_annotations
x_positions = np.arange(len(errors))

plt.figure(figsize=(14, 8))

for i, annotation in enumerate(annotations):
    plt.bar(
        x_positions + i * bar_width, 
        grouped_data[annotation], 
        width=bar_width, 
        label=annotation, 
    )


x_labels = ['Error 1', 'Error 2', 'Error 3']
plt.xticks(x_positions, x_labels, fontsize=14)

plt.ylabel(f"Frequency of {annotation_name}", fontsize=18)
plt.legend(title=annotation_name, title_fontsize=12, bbox_to_anchor=(0.77, 0.98), loc='upper left', fontsize=12, handleheight=2, handlelength=2, borderaxespad=0.1)

plt.tight_layout()
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.family': 'serif'})

plt.savefig(f'{annotation_name.lower()}_annotations_histogram_bw_l.png', format='png', dpi=300)
print("Plot saved.")

