import os
from joblib import load, dump
import pandas as pd


def merge_and_sort_joblib(folder_path):

    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in files if f.endswith('.joblib')]

    data = []
    for file in files:
        data.append(load(file))

    merged_data = [item for sublist in data for item in sublist]
    sorted_list = sorted(merged_data, key=lambda x: x['rew'], reverse=True)

    return sorted_list


folder_path = input("Please enter the path to the folder containing the joblib files: ")
sorted_data = merge_and_sort_joblib(folder_path)

output_path = input("Please enter the name of the new joblib file to save the sorted data: ")
dump(sorted_data, output_path)

# Save csv file
df = pd.DataFrame(sorted_data)
output_csv_path = input("Please enter the name of the new csv file to save the sorted data: ")
df.to_csv(output_csv_path, index=False)
