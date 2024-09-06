import pandas as pd
import ast
import math
from tqdm import tqdm

def flatten_nested_dict(d):
    flattened_dict = {}
    for key, nested_dict in d.items():
        flattened_dict.update(nested_dict)
    return flattened_dict

def is_valid_entry(flattened_dict):
    for key, value in flattened_dict.items():
        if value == 100.0 or value == 0.0:
            return False
    return True

def split_dataframe_to_excel(df, max_rows=1000000, output_prefix='output'):
    num_files = math.ceil(len(df) / max_rows)

    for i in tqdm(range(num_files), desc="Saving Excel files"):
        start_idx = i * max_rows
        end_idx = min((i + 1) * max_rows, len(df))

        df_subset = df.iloc[start_idx:end_idx]

        output_file = f"{output_prefix}_{i + 1}.xlsx"
        df_subset.to_excel(output_file, index=False)
        print(f"Saved {output_file}")

def process_reward(reward):
    if isinstance(reward, str):
        try:
            reward_dict = ast.literal_eval(reward)
            if isinstance(reward_dict, dict):
                return next(iter(reward_dict.values()))
        except:
            pass
    return reward

# Main process
file_path = input('Enter the file path: ')
data = pd.read_csv(file_path)

# Display the original data length
original_length = len(data)
print(f'Original data length: {original_length}')

# Flatten the nested dictionary and filter invalid entries for 'Specs'
print("Processing 'Specs' column...")
tqdm.pandas(desc="Flattening 'Specs'")
data['Valid_Specs'] = data['Specs'].progress_apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))
data = data[data['Valid_Specs'].apply(is_valid_entry)]

# Convert the valid flattened dictionaries into DataFrame columns for 'Specs'
specs_df = pd.DataFrame(data['Valid_Specs'].tolist())

# Expand 'Parameters' column directly into DataFrame columns
print("Processing 'Parameters' column...")
tqdm.pandas(desc="Expanding 'Parameters'")
params_df = data['Parameters'].progress_apply(ast.literal_eval).apply(pd.Series)

# Process the 'Reward' column
print("Processing 'Reward' column...")
tqdm.pandas(desc="Processing 'Reward'")
data['Reward'] = data['Reward'].progress_apply(process_reward)

# Combine the new columns with the original DataFrame (excluding the original 'Specs' and 'Valid_Specs' columns)
print("Combining processed data...")
expanded_data = pd.concat([data.drop(columns=['Specs', 'Valid_Specs', 'Parameters']), specs_df, params_df], axis=1)

# Generate new file path prefix
new_file_prefix = file_path.replace('.csv', '_format')

# Split and save to multiple Excel files
split_dataframe_to_excel(expanded_data, max_rows=1000000, output_prefix=new_file_prefix)

valid_length = len(expanded_data)
print(f'Total valid entries: {valid_length}')

# Optionally, to show how many entries were dropped
dropped_entries = original_length - valid_length
print(f'Entries dropped: {dropped_entries}')