import pandas as pd
from joblib import load


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

def calculate_and_sort_by_fom(df, formula, params):
    try:
        # Generate random positive float values for each parameter
        random_values = {param: np.random.rand() for param in params}
        # Test formula calculation with random values
        test_value = eval(formula, {}, random_values)
        if isinstance(test_value, float):
            # Calculate FoM for each row in DataFrame
            df['FoM'] = df.apply(lambda row: eval(formula, {}, row), axis=1)
            # Sort DataFrame by FoM
            sorted_df = df.sort_values(by='FoM', ascending=False)
            return sorted_df
        else:
            print("The formula did not evaluate to a float. Please check your formula.")
            return df
    except Exception as e:
        print(f"Error calculating or sorting by FoM: {e}")
        return df


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


def display_or_save_data(df, results):
    choice = input("Do you want to display the results or save them to a CSV file? (display/save): ")
    if choice.lower() == 'save':
        formula = input("Enter the FoM formula using the performance parameters: ")
        # Extract parameters from the results DataFrame
        params = results.columns
        # Ensure the formula contains all parameters
        if all(param in formula for param in params):
            sorted_df = calculate_and_sort_by_fom(df, formula, params)
            file_path = input("Enter the file path to save the CSV: ")
            sorted_df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        else:
            print("The formula must include all performance parameters.")
    elif choice.lower() == 'display':
        print(df.to_string())
    else:
        print("Invalid option. Please choose 'display' or 'save'.")


# Main script
if __name__ == "__main__":
    joblib_file = input("Joblib file path: ")
    df, results = load_data(joblib_file)
    summarize_data(results)
    filter_data(df, results)
