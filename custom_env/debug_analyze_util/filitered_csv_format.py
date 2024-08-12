import pandas as pd
import ast
from openpyxl import load_workbook
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


def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None,
                       truncate_sheet=False, **to_excel_kwargs):
    if not os.path.isfile(filename):
        df.to_excel(filename, sheet_name=sheet_name, startrow=startrow if startrow is not None else 0,
                    **to_excel_kwargs)
    else:
        writer = pd.ExcelWriter(filename, engine='openpyxl', mode='a')
        writer.book = load_workbook(filename)

        if startrow is None and sheet_name in writer.book.sheetnames:
            startrow = writer.book[sheet_name].max_row

        if truncate_sheet and sheet_name in writer.book.sheetnames:
            idx = writer.book.sheetnames.index(sheet_name)
            writer.book.remove(writer.book.worksheets[idx])
            writer.book.create_sheet(sheet_name, idx)

        writer.sheets = {ws.title: ws for ws in writer.book.worksheets}

        if startrow is None:
            startrow = 0

        df.to_excel(writer, sheet_name, startrow=startrow, **to_excel_kwargs)

        writer.save()


def main():
    file_path = input('Enter the file path: ')
    chunksize = 10000 # Number of rows to process at a time
    total_processed = 0
    total_valid = 0

    new_file_path = file_path.replace('.csv', '_format.xlsx')

    # Get total number of rows in the CSV file
    total_rows = sum(1 for line in open(file_path)) - 1  # Subtract 1 for the header
    print(f'Total rows in CSV: {total_rows}')

    for chunk_number, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        print(f'Processing chunk {chunk_number + 1}...')

        processed_chunk = process_chunk(chunk)
        total_processed += len(chunk)
        total_valid += len(processed_chunk)

        if chunk_number == 0:
            # First chunk, include header
            processed_chunk.to_excel(new_file_path, index=False)
        else:
            # Append to existing file without header
            append_df_to_excel(new_file_path, processed_chunk, header=False, index=False)

        print(f'Processed {total_processed} rows, Valid entries so far: {total_valid}')

    print(f'Total processed entries: {total_processed}')
    print(f'Total valid entries: {total_valid}')
    print(f'Entries dropped: {total_processed - total_valid}')
    print(f'Results saved to {new_file_path}')


if __name__ == "__main__":
    main()
