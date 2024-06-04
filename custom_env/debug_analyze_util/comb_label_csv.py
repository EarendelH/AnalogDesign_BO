import pandas as pd


def process_csv(file_path):
    """
    Load and process a CSV file.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the processed data.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    print(f"Loaded {data.shape[0]} rows from {file_path}.")

    # Remove duplicate rows based on the 'Directory Name' column
    data = data.drop_duplicates(subset='Directory Name')
    print(f"Rows after removing duplicates: {data.shape[0]}.")

    # Extract timestamps from the 'Directory Name' and convert to integer for sorting
    data['Timestamp'] = data['Directory Name'].str.extract(r'(\d{17})').astype(int)

    # Sort the DataFrame by the extracted timestamp
    data.sort_values('Timestamp', inplace=True)

    # Remove the auxiliary 'Timestamp' column as it is no longer needed after sorting
    data.drop('Timestamp', axis=1, inplace=True)

    return data


# Example usage: Replace 'path_to_csv_file' with the actual path to your CSV file
csv_file_path = 'label.csv'  # e.g., 'path_to_csv_file.csv'
processed_data = process_csv(csv_file_path)

# Save the processed DataFrame to a new CSV file
processed_data.to_csv('combined_label.csv', index=False)
