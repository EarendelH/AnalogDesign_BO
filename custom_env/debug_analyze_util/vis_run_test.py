import os
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Function to validate entries using specific criteria
def is_valid_entry(input_dict):
    flattened_dict = {k: v for d in input_dict.values() for k, v in d.items()}
    valid_values = [value for value in flattened_dict.values() if value not in [0.0, 100.0]]
    return len(valid_values)


# Function to scan directories and sort them by timestamp
def scan_and_sort_directories(base_path):
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sorted_dirs = sorted(
        (d for d in dirs if d.startswith("tmp_")),
        key=lambda x: x[4:18]  # Extracting the timestamp part from the folder name
    )
    return sorted_dirs


# Function to label directories and write labels to a CSV file
def label_directories(base_path, sorted_dirs):
    labels = {}
    with open('directory_labels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Directory Name', 'Label'])
        for dir_name in sorted_dirs:
            dir_path = os.path.join(base_path, dir_name)
            subdirs = os.listdir(dir_path)
            if 'Region.raw' in subdirs and len(subdirs) == 1:
                labels[dir_name] = 0  # Only contains Region.raw
            else:
                result_path = os.path.join(dir_path, 'result.pkl')
                if os.path.exists(result_path):
                    with open(result_path, 'rb') as file:
                        result_dict = pickle.load(file)
                    labels[dir_name] = is_valid_entry(result_dict)  # Valid result
                else:
                    labels[dir_name] = 0  # Default label if conditions are not met
            writer.writerow([dir_name, labels[dir_name]])
            print(f"Directory {dir_name} labeled as {labels[dir_name]}")
    return labels


# Function to plot the distribution of labels
def plot_label_distribution(labels, group_size=10000):
    label_values = list(labels.values())
    max_label = max(label_values) if label_values else 0
    colors = LinearSegmentedColormap.from_list("gradient", ["#FFFFFF", "#1C4E87"], N=max_label + 1)

    num_groups = (len(label_values) + group_size - 1) // group_size
    results = []

    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, len(label_values))
        current_group = label_values[start:end]
        counts = {label: 0 for label in range(max_label + 1)}
        for label in current_group:
            counts[label] = counts.get(label, 0) + 1
        proportions = {key: value / len(current_group) * 100 for key, value in counts.items()}
        results.append(proportions)

    fig, ax = plt.subplots()
    base = np.zeros(num_groups)
    for label in range(max_label + 1):
        heights = [result.get(label, 0) for result in results]
        ax.bar(np.arange(num_groups) * group_size, heights, bottom=base, width=group_size, label=f'Label {label}',
               color=colors(label))
        base = np.add(base, heights)

    ax.set_xlabel('Directory Number Range')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Distribution of Labels per Group of 10000 Directories')
    ax.set_xticks(np.arange(0, group_size * num_groups, group_size))
    ax.set_xticklabels([f"{x}-{x + group_size - 1}" for x in range(0, len(label_values), group_size)])

    ax.legend()
    plt.show()


# Main code
if __name__ == "__main__":
    base_path = input("Enter the path to the directory: ")
    sorted_dirs = scan_and_sort_directories(base_path)
    labels = label_directories(base_path, sorted_dirs)
    # plot_label_distribution can be modified to read from 'directory_labels.csv' if running separately
    plot_label_distribution(labels)
