import pandas as pd
import ast
import math
from tqdm import tqdm
from collections import Counter


def flatten_nested_dict(d):
    flattened_dict = {}
    for key, nested_dict in d.items():
        flattened_dict.update(nested_dict)
    return flattened_dict


def get_invalid_reasons(flattened_dict):
    invalid_keys = []
    for key, value in flattened_dict.items():
        if value == 100.0 or value == 0.0:
            invalid_keys.append(key)
    return invalid_keys

def is_valid_entry(flattened_dict):
    return len(get_invalid_reasons(flattened_dict)) == 0

def generate_invalid_stats(data):
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
    num_files = math.ceil(len(df) / max_rows)

    for i in range(num_files):
        start_idx = i * max_rows
        end_idx = min((i + 1) * max_rows, len(df))

        df_subset = df.iloc[start_idx:end_idx]

        output_file = f"{output_prefix}_{i + 1}.xlsx"
        df_subset.to_excel(output_file, index=False)
        print(f"Saved {output_file}")


# Main process
file_path = input('Enter the file path: ')
need_filter = input('Do you need to filter invalid entries? (y/n): ').lower() == 'y'

data = pd.read_csv(file_path)
original_length = len(data)
print(f'Original data length: {original_length}')

print("Processing 'Specs' column...")
tqdm.pandas(desc="Flattening Specs")
data['Valid_Specs'] = data['Specs'].progress_apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))

print("Generating invalid entries statistics...")
stats_df = generate_invalid_stats(data)
stats_file = file_path.replace('.csv', '_invalid_stats.xlsx')
stats_df.to_excel(stats_file, index=False)
print(f"Invalid entry statistics saved to: {stats_file}")

if need_filter:
    print("Filtering invalid entries...")
    data = data[data['Valid_Specs'].apply(is_valid_entry)]
    output_suffix = '_format'
else:
    print("Keeping all entries...")
    output_suffix = '_unfiltered'

print("Processing 'Parameters' column...")
tqdm.pandas(desc="Expanding Parameters")
params_df = data['Parameters'].progress_apply(ast.literal_eval).apply(pd.Series)

print("Expanding Specs into columns...")
specs_df = data['Valid_Specs'].apply(pd.Series)

print("Combining data...")
expanded_data = pd.concat([data.drop(columns=['Specs', 'Valid_Specs', 'Parameters']),
                           specs_df, params_df], axis=1)

new_file_prefix = file_path.replace('.csv', output_suffix)

split_dataframe_to_excel(expanded_data, max_rows=1000000, output_prefix=new_file_prefix)

valid_length = len(expanded_data)
print(f'Total entries in output: {valid_length}')

if need_filter:
    dropped_entries = original_length - valid_length
    print(f'Entries dropped: {dropped_entries}')