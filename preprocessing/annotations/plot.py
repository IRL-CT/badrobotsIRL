import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

annotation_list = [
     #"prompt", "repeated prompt", "more specific/longer prompt", "swapping terms in prompt", "simpler prompt",
     #"slower", "demanding tone", "interrogative tone", "filler words", "humor",
     "amusement", "confusion", "frustration",
     "quitting", "looking at tablet",
    # "looking at door",
]

# color_palette = cm.get_cmap('tab20', len(annotation_list))

# annotation_colors = {annotation_list[i]: color_palette(i) for i in range(len(annotation_list))}
annotation_colors_bw = ["#D3D3D3", "#A9A9A9", "#808080", "#505050", "#303030",
                        #"#D3D3D3", "#A9A9A9", "#808080", "#505050", "#303030",
                        # "#D3D3D3", "#A9A9A9", "#808080", "#505050", "#303030",
                        ]

data = pd.read_csv('sorted_nodbot_annotations.csv')

data['labels'] = data['labels'].str.split(", ").apply(lambda x: [label for label in x if label in annotation_list])

data = data.explode('labels')

grouped_data = data.groupby(['error', 'labels']).size().unstack(fill_value=0)

grouped_data = grouped_data.reindex(columns=annotation_list, fill_value=0)

errors = grouped_data.index
annotations = grouped_data.columns
num_annotations = len(annotations)
bar_width = 0.8 / num_annotations
x_positions = np.arange(len(errors))

plt.figure(figsize=(16, 8))
# for i, annotation in enumerate(annotations):
#     plt.bar(
#         x_positions + i * bar_width, 
#         grouped_data[annotation], 
#         width=bar_width, 
#         label=annotation, 
#         color=annotation_colors_bw[i]
#     )

hatch_patterns = [' ', '//', ' ', '\\\\', ' ', '///',
                  '\\\\\\', ' ', '////', ' ', '\\\\\\\\',
                #   '/', '\\', '/////', '\\\\\\\\\\', ' '
                  ]

for i, annotation in enumerate(annotations):
    plt.bar(
        x_positions + i * bar_width, 
        grouped_data[annotation], 
        width=bar_width, 
        label=annotation, 
        color=annotation_colors_bw[i],
        #hatch=hatch_patterns[i % len(hatch_patterns)]
    )


x_labels = ['Participant alerts','Error 1', 'Error 2', 'Error 3', 'Researcher called']
plt.xticks(x_positions, x_labels, fontsize=12)

plt.title("Distribution of Participant Nonverbal Response Types Across Successive Robot Errors", fontsize=16)
plt.xlabel("Participant and Robot Interaction Points", fontsize=14)
plt.ylabel("Frequency of Nonverbal Response Type", fontsize=14)
plt.legend(title="Common Participant Nonverbal Response Types", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, handleheight=3, handlelength=3, borderaxespad=0.1)
plt.tight_layout()

plt.savefig('nonverbal_annotations_histogram_bw.png', format='png', dpi=300)
print("Plot saved.")

plt.show()