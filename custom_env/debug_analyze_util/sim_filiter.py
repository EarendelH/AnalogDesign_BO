import pandas as pd
from joblib import load
import numpy

def load_data(joblib_file_path):
    # Load data from the specified joblib file
    data = load(joblib_file_path)
    # Convert loaded data into a DataFrame
    # Assuming each item in 'data' is a dictionary with 'result' and 'param'
    df = pd.DataFrame(data)
    # Extracting results into a separate DataFrame for easier manipulation
    results = pd.json_normalize(df['result'])
    return df, results


def summarize_data(df):
    # Summarize performance parameters: print max and min values, and show number of data groups
    print("\nPerformance Parameters Summary:")
    for column in df.columns:
        # Skip summarization for non-numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"{column}: Min = {df[column].min()}, Max = {df[column].max()}")
    print(f"Number of data groups: {len(df)}")


def filter_data(df, results):
    # Display a list of performance parameters for the user to choose from
    print("\nPlease choose a performance parameter to filter by:")
    for i, column in enumerate(results.columns, 1):
        print(f"{i}. {column}")

    choice = int(input("Enter the performance parameter number: ")) - 1
    param = results.columns[choice]

    # Display selected performance parameter's max and min values
    print(f"{param}: Min = {results[param].min()}, Max = {results[param].max()}")

    # User input for filter range
    range_input = input(f"Enter the range for {param} (e.g., 1-10 or -10 or 1-): ")
    min_val, max_val = (None if part == '' else float(part) for part in range_input.split('-'))

    # Filter data based on the range
    condition = results[param].between(min_val if min_val is not None else results[param].min(),
                                       max_val if max_val is not None else results[param].max())
    filtered_results = results[condition]
    filtered_df = df[condition]

    # Summarize the filtered data
    summarize_data(filtered_results)

    # Ask the user if they want to continue filtering
    continue_filter = input("Do you want to continue filtering? (y/n): ")
    if continue_filter.lower() == 'y':
        filter_data(filtered_df, filtered_results)  # Recursive call to continue filtering
    else:
        display_or_save_data(filtered_df)


def calculate_fom(row):
    return (row['phaseMargin']*numpy.log10(row['gainBandWidth'])*row['powerSupplyRejectionRatio']) / row['pwr']


def display_or_save_data(df):
    choice = input("Do you want to display the results or save them to a CSV file? (display/save): ")
    if choice.lower() == 'display':
        print(df.to_string())
    elif choice.lower() == 'save':
        # 在保存之前计算FoM并排序
        df['FoM'] = df.apply(calculate_fom, axis=1)
        df_sorted = df.sort_values(by='FoM', ascending=False)

        file_path = input("Enter the file path to save the CSV: ")
        df_sorted.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("Invalid option. Please choose 'display' or 'save'.")


# Main script
if __name__ == "__main__":
    joblib_file = input("Joblib file path: ")
    df, results = load_data(joblib_file)
    summarize_data(results)
    filter_data(df, results)