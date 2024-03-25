import os
from joblib import load, dump
import pandas as pd


def sort_joblib(file_path):

    data = load(file_path)

    sorted_list = sorted(data, key=lambda x: x['rew'], reverse=True)

    return sorted_list


folder_path = input("Please enter the path to the joblib files: ")
sorted_data = sort_joblib(folder_path)

output_path = input("Please enter the name of the new joblib file to save the sorted data: ")
dump(sorted_data, output_path)

# Save csv file
df = pd.DataFrame(sorted_data)
output_csv_path = input("Please enter the name of the new csv file to save the sorted data: ")
df.to_csv(output_csv_path, index=False)
