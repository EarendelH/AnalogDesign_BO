import pandas as pd
import os


def process_csv_files(directory_path):
    """
    Load and process all CSV files within a specified directory.

    Args:
    directory_path (str): The path to the directory containing CSV files.

    Returns:
    pd.DataFrame: A DataFrame containing the combined and processed data from all files.
    """
    # Scan the directory for all CSV files
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    print(f"Found {len(file_paths)} CSV files in the directory.")

    # Load each CSV file into a DataFrame and store it in a list
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
        print(f"Loaded {df.shape[0]} rows from {path}.")

    # Concatenate all DataFrames into a single DataFrame
    combined_data = pd.concat(data_frames, ignore_index=True)
    print(f"Total combined rows after merging: {combined_data.shape[0]}.")

    # Remove duplicate rows based on the 'Directory Name' column
    combined_data = combined_data.drop_duplicates(subset='Directory Name')
    print(f"Rows after removing duplicates: {combined_data.shape[0]}.")

    # Extract timestamps from the 'Directory Name' and convert to integer for sorting
    combined_data['Timestamp'] = combined_data['Directory Name'].str.extract(r'(\d{17})').astype(int)

    # Sort the DataFrame by the extracted timestamp
    combined_data.sort_values('Timestamp', inplace=True)

    # Remove the auxiliary 'Timestamp' column as it is no longer needed after sorting
    combined_data.drop('Timestamp', axis=1, inplace=True)

    return combined_data


# Example usage: Replace 'path_to_directory' with the actual path to your directory
directory_path = input('Enter the directory path: ')  # e.g., 'path_to_directory'
final_data = process_csv_files(directory_path)

# Save the final combined and sorted DataFrame to a CSV file
final_data.to_csv('final_combined_output.csv', index=False)