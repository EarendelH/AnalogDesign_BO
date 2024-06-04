import pandas as pd
import sys


# Function to convert strings with metric prefixes to float
def convert_to_float(value):
    suffixes = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'M': 1e6}
    if isinstance(value, str) and value[-1] in suffixes:
        # Calculate the scale based on the suffix and convert the number
        scale = suffixes[value[-1]]
        number = float(value[:-1])
        return number * scale
    return value


def process_excel(filepath):
    # Load the Excel file
    df = pd.read_excel(filepath)

    # Remove the first and second column (folder name)
    df = df.iloc[:, 2:]

    # Convert values with metric suffixes to float
    for col in df.columns[21:]:
        print(f'Converting column: {col}')
        df[col] = df[col].apply(convert_to_float)

    # Find all columns that start with 'nf_'
    nf_columns = [col for col in df.columns if col.startswith('nf_')]

    # Perform calculations and modify dataframe for each 'nf_' column
    for nf_col in nf_columns:
        print(f'Processing column: {nf_col}')
        prefix = nf_col.split('_')[1]  # Assume column format is 'nf_MX'
        w_col = f'w_{prefix}_per_finger'  # Corresponding width column
        new_col = f'w_{prefix}'  # New column name
        df[new_col] = df[nf_col] * df[w_col]  # Compute new column values
        df.drop(columns=[nf_col, w_col], inplace=True)  # Remove old columns

    # Save the modified dataframe to a new Excel file
    output_path = filepath.replace('.xlsx', '_processed.xlsx')
    df.to_excel(output_path, index=False)
    print(f'Processed file saved as: {output_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the file path as an argument.")
    else:
        path = sys.argv[1]
        process_excel(path)
