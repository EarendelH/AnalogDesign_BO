import os
from joblib import load, dump
import pandas as pd


def merge_and_sort_joblib(folder_path):
    all_data = []

    for file in os.listdir(folder_path):
        if file.endswith('.joblib'):
            file_path = os.path.join(folder_path, file)
            data = load(file_path)
            all_data.append(data)

    # Sort all_data by the 'rew' key

    return all_data


folder_path = input("Please enter the path to the folder containing the joblib files: ")
sorted_data = merge_and_sort_joblib(folder_path)

output_path = input("Please enter the name of the new joblib file to save the sorted data: ")
dump(sorted_data, output_path)

# Save csv file
df = pd.DataFrame(sorted_data)
output_csv_path = input("Please enter the name of the new csv file to save the sorted data: ")
df.to_csv(output_csv_path, index=False)
