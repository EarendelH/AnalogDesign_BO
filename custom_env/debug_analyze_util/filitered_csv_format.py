import pandas as pd
import ast


def flatten_nested_dict(d):
    flattened_dict = {}
    for key, nested_dict in d.items():
        flattened_dict.update(nested_dict)
    return flattened_dict


def is_valid_entry(flattened_dict):
    for key, value in flattened_dict.items():
        if (key.startswith('Trans') or key.startswith('DC')) and value == 100.0:
            return False
        if not (key.startswith('Trans') or key.startswith('DC')) and value == 0.0:
            return False
    return True


file_path = input('Enter the file path: ')
data = pd.read_csv(file_path)

# Display the original data length
original_length = len(data)
print(f'Original data length: {original_length}')

# Flatten the nested dictionary and filter invalid entries
data['Valid_Specs'] = data['Specs'].apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))
# data = data[data['Valid_Specs'].apply(is_valid_entry)]

# Convert the valid flattened dictionaries into DataFrame columns
result_df = data['Valid_Specs'].apply(pd.Series)

# Combine the new columns with the original DataFrame (excluding the original 'Specs' column)
expanded_data = pd.concat([data.drop(columns=['Specs', 'Valid_Specs']), result_df], axis=1)

# Save to a new CSV file and print the count of valid entries
new_file_path = file_path.replace('.csv', '_format.xlsx')
expanded_data.to_excel(new_file_path, index=False)
valid_length = len(expanded_data)
print(f'Total valid entries: {valid_length}')

# Optionally, to show how many entries were dropped
dropped_entries = original_length - valid_length
print(f'Entries dropped: {dropped_entries}')
