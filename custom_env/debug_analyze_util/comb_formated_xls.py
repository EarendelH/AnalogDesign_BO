import pandas as pd
import numpy as np
import os


def combine_sort_xls(directory_path):
    """
    Load and process all Excel files within a specified directory.

    Args:
    directory_path (str): The path to the directory containing Excel files.

    Returns:
    pd.DataFrame: A DataFrame containing the combined and processed data from all files.
    """
    # Scan the directory for all Excel files with a .xlsx extension
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.xlsx')]
    print(f"Found {len(file_paths)} Excel files in the directory.")

    # Load each Excel file into a DataFrame and store it in a list
    data_frames = []
    for path in file_paths:
        df = pd.read_excel(path)
        data_frames.append(df)
        print(f"Loaded {df.shape[0]} rows from {path}.")

    # Concatenate all DataFrames into a single DataFrame
    combined_data = pd.concat(data_frames, ignore_index=True)
    print(f"Total combined rows after merging: {combined_data.shape[0]}.")

    # Remove duplicate rows based on the 'Folder Name' column
    combined_data = combined_data.drop_duplicates(subset='Folder Name')
    print(f"Rows after removing duplicates: {combined_data.shape[0]}.")

    # Extract timestamps from the 'Folder Name' and convert to integer for sorting
    combined_data['Timestamp'] = combined_data['Folder Name'].str.extract(r'(\d{14})').astype(int)

    # Sort the DataFrame by the extracted timestamp
    combined_data.sort_values('Timestamp', inplace=True)

    # Remove the auxiliary 'Timestamp' column as it is no longer needed after sorting
    combined_data.drop('Timestamp', axis=1, inplace=True)

    return combined_data


# Example usage: Replace 'path_to_directory' with the actual path to your directory
directory_path = input('Enter the directory path: ')  # e.g., 'path_to_directory
final_data = combine_sort_xls(directory_path)

# Save the final combined and sorted DataFrame to an Excel file
final_data.to_excel('final_combined_output.xlsx', index=False)
