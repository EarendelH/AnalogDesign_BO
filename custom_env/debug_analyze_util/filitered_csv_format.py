import pandas as pd
import ast
import math
import argparse
from tqdm import tqdm
from collections import Counter


def flatten_nested_dict(d):
    """
    Flatten a nested dictionary by merging all sub-dictionaries into a single dictionary

    Args:
        d: A nested dictionary structure

    Returns:
        A flattened dictionary
    """
    flattened_dict = {}
    for key, nested_dict in d.items():
        flattened_dict.update(nested_dict)
    return flattened_dict


def get_invalid_reasons(flattened_dict):
    """
    Find keys with invalid values (0.0 or 100.0)

    Args:
        flattened_dict: A flattened dictionary of specs

    Returns:
        List of keys with invalid values
    """
    invalid_keys = []
    for key, value in flattened_dict.items():
        if value == 100.0 or value == 0.0:
            invalid_keys.append(key)
    return invalid_keys


def is_valid_entry(flattened_dict):
    """
    Check if a dictionary contains any invalid values

    Args:
        flattened_dict: A flattened dictionary of specs

    Returns:
        Boolean indicating whether the entry is valid
    """
    return len(get_invalid_reasons(flattened_dict)) == 0


def generate_invalid_stats(data):
    """
    Generate statistics about invalid entries

    Args:
        data: DataFrame with Valid_Specs column

    Returns:
        DataFrame with statistics about invalid parameters
    """
    invalid_combinations = []

    for specs_dict in tqdm(data['Valid_Specs'], desc="Analyzing invalid entries"):
        invalid_keys = get_invalid_reasons(specs_dict)
        if invalid_keys:
            invalid_combinations.append(', '.join(sorted(invalid_keys)))

    stats_counter = Counter(invalid_combinations)

    stats_df = pd.DataFrame(list(stats_counter.items()),
                            columns=['Invalid_Parameters', 'Frequency'])
    stats_df = stats_df.sort_values('Frequency', ascending=False)

    return stats_df


def split_dataframe_to_excel(df, max_rows=1000000, output_prefix='output'):
    """
    Split a large DataFrame into multiple Excel files

    Args:
        df: DataFrame to split
        max_rows: Maximum rows per file
        output_prefix: Prefix for output files

    Returns:
        None, files are saved to disk
    """
    num_files = math.ceil(len(df) / max_rows)

    for i in range(num_files):
        start_idx = i * max_rows
        end_idx = min((i + 1) * max_rows, len(df))

        df_subset = df.iloc[start_idx:end_idx]

        output_file = f"{output_prefix}_{i + 1}.xlsx"
        df_subset.to_excel(output_file, index=False)
        print(f"Saved {output_file}")


def parse_args():
    """
    Parse command line arguments

    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Process and format CSV data')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--filter', action='store_true', help='Filter invalid entries')

    return parser.parse_args()


def main():
    """
    Main function with command line or interactive input support
    """
    # Parse command line arguments
    args = parse_args()

    # Get input file path (from args or user input)
    file_path = args.input
    if file_path is None:
        file_path = input('Enter the file path: ')

    # Determine if filtering is needed (from args or user input)
    need_filter = args.filter
    if args.input is None:  # Only ask in interactive mode
        need_filter = input('Do you need to filter invalid entries? (y/n): ').lower() == 'y'

    # Load data
    data = pd.read_csv(file_path)
    original_length = len(data)
    print(f'Original data length: {original_length}')

    # Process specs column
    print("Processing 'Specs' column...")
    tqdm.pandas(desc="Flattening Specs")
    data['Valid_Specs'] = data['Specs'].progress_apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))

    # Generate statistics about invalid entries
    print("Generating invalid entries statistics...")
    stats_df = generate_invalid_stats(data)
    stats_file = file_path.replace('.csv', '_invalid_stats.xlsx')
    stats_df.to_excel(stats_file, index=False)
    print(f"Invalid entry statistics saved to: {stats_file}")

    # Filter data if requested
    if need_filter:
        print("Filtering invalid entries...")
        data = data[data['Valid_Specs'].apply(is_valid_entry)]
        output_suffix = '_format'
    else:
        print("Keeping all entries...")
        output_suffix = '_unfiltered'

    # Process parameters column
    print("Processing 'Parameters' column...")
    tqdm.pandas(desc="Expanding Parameters")
    params_df = data['Parameters'].progress_apply(ast.literal_eval).apply(pd.Series)

    # Expand specs into columns
    print("Expanding Specs into columns...")
    specs_df = data['Valid_Specs'].apply(pd.Series)

    # Combine all data
    print("Combining data...")
    expanded_data = pd.concat([data.drop(columns=['Specs', 'Valid_Specs', 'Parameters']),
                               specs_df, params_df], axis=1)

    # Create output files
    new_file_prefix = file_path.replace('.csv', output_suffix)
    split_dataframe_to_excel(expanded_data, max_rows=1000000, output_prefix=new_file_prefix)

    # Print summary
    valid_length = len(expanded_data)
    print(f'Total entries in output: {valid_length}')

    if need_filter:
        dropped_entries = original_length - valid_length
        print(f'Entries dropped: {dropped_entries}')


if __name__ == "__main__":
    main()