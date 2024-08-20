import os
from utils import calculate_and_save_average_metrics

gru_directory = "training/gru/"
lstm_directory = "training/lstm/"
transformer_directory = "training/transformer"

for filename in os.listdir(gru_directory):

    nested_directory = os.path.join(gru_directory, filename)

    calculate_and_save_average_metrics(nested_directory, f"{nested_directory}/all_metrics.txt")


for filename in os.listdir(lstm_directory):

    nested_directory = os.path.join(lstm_directory, filename)

    calculate_and_save_average_metrics(nested_directory, f"{nested_directory}/all_metrics.txt")


for filename in os.listdir(transformer_directory):

    nested_directory = os.path.join(transformer_directory, filename)

    calculate_and_save_average_metrics(nested_directory, f"{nested_directory}/all_metrics.txt")