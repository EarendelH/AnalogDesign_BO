import pandas as pd
import ast

file_path = input('Enter the file path: ')
data = pd.read_csv(file_path)

data['result'] = data['result'].apply(ast.literal_eval)

result_df = data['result'].apply(pd.Series)

expanded_data = pd.concat([data.drop(columns=['result']), result_df], axis=1)

new_file_path = file_path.replace('.csv', '_format.csv')

expanded_data.to_csv(new_file_path, index=False)
