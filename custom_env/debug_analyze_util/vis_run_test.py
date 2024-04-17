import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


# Function to validate entries using specific criteria
def is_valid_entry(input_dict):
    """Check validity of dictionary entries based on predefined rules."""

    flattened_dict = {k: v for d in input_dict.values() for k, v in d.items()}

    for key, value in flattened_dict.items():
        if (key.startswith('Trans') or key.startswith('DC')) and value == 1.0:
            return False
        if not (key.startswith('Trans') or key.startswith('DC')) and value == 0.0:
            return False
    return True


# Function to scan directories and sort them by timestamp
def scan_and_sort_directories(base_path):
    """Scan and sort directories based on timestamp in their names."""
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sorted_dirs = sorted(
        (d for d in dirs if d.startswith("tmp_")),
        key=lambda x: x[4:18]  # Extracting the timestamp part from the folder name
    )
    return sorted_dirs


# Function to label directories based on specific criteria
def label_directories(base_path, sorted_dirs):
    """Label directories based on the presence of specific files and subdirectories."""
    labels = {}
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
                if is_valid_entry(result_dict):
                    labels[dir_name] = 2  # Valid result
                else:
                    labels[dir_name] = 1  # Invalid result
            else:
                labels[dir_name] = 0  # Default label if conditions are not met
        print(f"Directory {dir_name} labeled as {labels[dir_name]}")
    return labels


# Function to plot the distribution of labels
def plot_label_distribution(labels, group_size=10000):
    """Plot the distribution of directory labels in groups of a specified size."""
    label_values = list(labels.values())
    num_groups = (len(label_values) + group_size - 1) // group_size
    results = []

    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, len(label_values))
        current_group = label_values[start:end]
        counts = {0: 0, 1: 0, 2: 0}
        for label in current_group:
            counts[label] = counts.get(label, 0) + 1
        proportions = {key: value / len(current_group) * 100 for key, value in counts.items()}
        results.append(proportions)

    # Plotting the distribution
    fig, ax = plt.subplots()
    colors = {'red': '#E41A1C', 'orange': '#FF7F00', 'blue': '#377EB8'}
    base = np.zeros(num_groups)
    for idx, (color, label) in enumerate(zip(['red', 'orange', 'blue'], [0, 1, 2])):
        heights = [result[label] for result in results]
        ax.bar(np.arange(num_groups) * group_size, heights, bottom=base, width=group_size, label=f'Label {label}', color=colors[color])
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
    base_path = input("Enter the path to the directory: ")  # User inputs the directory path
    sorted_dirs = scan_and_sort_directories(base_path)
    labels = label_directories(base_path, sorted_dirs)
    plot_label_distribution(labels)
