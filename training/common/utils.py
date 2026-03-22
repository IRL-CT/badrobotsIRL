import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_metrics_to_file(metrics, fold_no, model, sweep_name):

    """
    Creates a file of metrics  

    Requires:
        a dictionary containing metrics
        a path to the file where metrics will be saved
    """

    dir_path = f'training/{model}/{sweep_name}'
    
    os.makedirs(dir_path, exist_ok=True)
    
    file_path = f'{dir_path}/metrics_fold_{fold_no}.txt'

    print(f"Saving metrics for Fold {fold_no}: {metrics}")
    
    with open(file_path, 'w') as file:
        file.write(f"Metrics for {model} Sweep: {sweep_name}, Fold: {fold_no}\n")
        file.write("="*50 + "\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")


def plot_train_accuracy(train_accuracies, fold_no, model, sweep_name):

    """
    Creates a plot of train accuracy over epochs

    Requires:
        a list of training accuracies for each epoch
        the total number of epochs
        the fold number
        the name of the sweep
    """

    plt.clf()

    dir_path = f'training/{model}/{sweep_name}'
    os.makedirs(dir_path, exist_ok=True)
    
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Train Accuracy for {model} {sweep_name} - Fold {fold_no}')
    plt.legend()

    file_path = f'{dir_path}/{sweep_name}_train_accuracy_fold_{fold_no}.png'
    plt.savefig(file_path)
    plt.close()


def plot_val_accuracy(val_accuracies, fold_no, model, sweep_name):
    
    """
    Creates a plot of validation accuracy over epochs

    Requires:
        a list of validation accuracies for each epoch
        the total number of epochs
        the fold number
        the name of the sweep
    """

    plt.clf()

    dir_path = f'training/{model}/{sweep_name}'
    os.makedirs(dir_path, exist_ok=True)

    epochs = range(1, len(val_accuracies) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy for {model} {sweep_name} - Fold {fold_no}')
    plt.legend()

    file_path = f'{dir_path}/val_accuracy_fold_{fold_no}.png'
    plt.savefig(file_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, fold_no, model, sweep_name):

    """
    Creates a confusion matrix and saves in directory

    Requires:
        an array of true labels.
        an array of predicted labels.
        the fold number.
        the name of the sweep.
    """

    plt.clf()

    dir_path = f'training/{model}/{sweep_name}'
    os.makedirs(dir_path, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{sweep_name} Confusion Matrix for {model} {sweep_name} - Fold {fold_no}')

    file_path = f'{dir_path}/{sweep_name}_confusion_matrix_fold_{fold_no + 1}.png'
    plt.savefig(file_path)
    plt.close()


def calculate_and_save_average_metrics(input_directory, output_filepath):
    """
    Calculates average metrics across multiple files in a directory and saves them to a file.

    Requires:
        the directory containing the metrics files.
        the path to the output file where the average metrics will be saved.
    """
    all_metrics = defaultdict(list)
    file_count = 0

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_directory, filename)

            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    match = re.match(r'(\w+): ([\d.]+)', line)
                    if match:
                        metric_name = match.group(1)
                        metric_value = float(match.group(2))
                        all_metrics[metric_name].append(metric_value)
            file_count += 1

    if file_count == 0:
        print("No metrics files found.")
        return

    average_metrics = {key: sum(values)/file_count for key, values in all_metrics.items()}

    with open(output_filepath, 'w') as file:
        file.write("Average Metrics\n")
        file.write("="*50 + "\n")
        for key, value in average_metrics.items():
            file.write(f"{key}: {value}\n")
    
    print(f"Average metrics have been saved to {output_filepath}")
