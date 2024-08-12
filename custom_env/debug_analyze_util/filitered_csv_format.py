import pandas as pd
import ast
import openpyxl
import os


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


def process_chunk(chunk):
    # Flatten the nested dictionary and filter invalid entries for 'Specs'
    chunk['Valid_Specs'] = chunk['Specs'].apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))
    chunk = chunk[chunk['Valid_Specs'].apply(is_valid_entry)]

    # Convert the valid flattened dictionaries into DataFrame columns for 'Specs'
    specs_df = chunk['Valid_Specs'].apply(pd.Series)

    # Expand 'Parameters' column directly into DataFrame columns
    params_df = chunk['Parameters'].apply(ast.literal_eval).apply(pd.Series)

    # Combine the new columns with the original DataFrame (excluding the original 'Specs' and 'Valid_Specs' columns)
    return pd.concat([chunk.drop(columns=['Specs', 'Valid_Specs', 'Parameters']), specs_df, params_df], axis=1)


def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None, **to_excel_kwargs):
    # Excel file doesn't exist - saving and exiting
    if not os.path.isfile(filename):
        df.to_excel(filename, sheet_name=sheet_name, **to_excel_kwargs)
        return

    # Excel file exists - append without writing the header
    book = openpyxl.load_workbook(filename)
    sheet = book[sheet_name]
    rows = dataframe_to_rows(df, index=False, header=False)
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            sheet.cell(row=sheet.max_row + r_idx, column=c_idx, value=value)

    book.save(filename)


def dataframe_to_rows(df, index=False, header=True):
    if header:
        yield df.columns.tolist()
    for idx, row in df.iterrows():
        yield row.tolist()


def main():
    file_path = input('Enter the file path: ')
    chunksize = 10000  # Adjust this value based on your system's memory capacity
    total_processed = 0
    total_valid = 0

    new_file_path = file_path.replace('.csv', '_format.xlsx')

    # Get total number of rows
    total_rows = sum(1 for line in open(file_path)) - 1  # Subtract header row
    print(f'Total rows in CSV: {total_rows}')

    for chunk_number, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        print(f'Processing chunk {chunk_number + 1}...')

        processed_chunk = process_chunk(chunk)
        total_processed += len(chunk)
        total_valid += len(processed_chunk)

        if chunk_number == 0:
            # First chunk, write with header
            processed_chunk.to_excel(new_file_path, index=False)
        else:
            # Subsequent chunks, append without header
            append_df_to_excel(new_file_path, processed_chunk, index=False)

        print(f'Processed {total_processed} rows, Valid entries so far: {total_valid}')

    print(f'Total processed entries: {total_processed}')
    print(f'Total valid entries: {total_valid}')
    print(f'Entries dropped: {total_processed - total_valid}')
    print(f'Results saved to {new_file_path}')


if __name__ == "__main__":
    main()