import pandas as pd
import ast


def flatten_nested_dict(d):
    flattened_dict = {}
    for key, nested_dict in d.items():
        flattened_dict.update(nested_dict)
    return flattened_dict


file_path = input('Enter the file path: ')
data = pd.read_csv(file_path)

data['Specs'] = data['Specs'].apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))

result_df = data['Specs'].apply(pd.Series)

expanded_data = pd.concat([data.drop(columns=['Specs']), result_df], axis=1)

new_file_path = file_path.replace('.csv', '_format.csv')
expanded_data.to_csv(new_file_path, index=False)
