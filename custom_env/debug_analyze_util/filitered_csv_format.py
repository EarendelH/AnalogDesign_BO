import pandas as pd
import ast
import openpyxl
import os

MAX_ROWS_PER_SHEET = 1000000  # Slightly less than the Excel limit to be safe


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
    chunk['Valid_Specs'] = chunk['Specs'].apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))
    chunk = chunk[chunk['Valid_Specs'].apply(is_valid_entry)]
    specs_df = chunk['Valid_Specs'].apply(pd.Series)
    params_df = chunk['Parameters'].apply(ast.literal_eval).apply(pd.Series)
    return pd.concat([chunk.drop(columns=['Specs', 'Valid_Specs', 'Parameters']), specs_df, params_df], axis=1)


def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None, **to_excel_kwargs):
    if not os.path.isfile(filename):
        df.to_excel(filename, sheet_name=sheet_name, **to_excel_kwargs)
        return

    book = openpyxl.load_workbook(filename)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    writer.book = book

    if sheet_name not in writer.book.sheetnames:
        writer.book.create_sheet(sheet_name)
    sheet = writer.book[sheet_name]

    if sheet.max_row >= MAX_ROWS_PER_SHEET:
        sheet_count = sum(1 for s in writer.book.sheetnames if s.startswith(sheet_name))
        new_sheet_name = f"{sheet_name}_{sheet_count + 1}"
        writer.book.create_sheet(new_sheet_name)
        sheet = writer.book[new_sheet_name]
        startrow = 0
    else:
        startrow = sheet.max_row if startrow is None else startrow

    df.to_excel(writer, sheet_name=sheet.title, startrow=startrow, **to_excel_kwargs)
    writer.save()
    writer.close()


def main():
    file_path = input('Enter the file path: ')
    chunksize = 10000  # Adjust based on your system's memory capacity
    total_processed = 0
    total_valid = 0

    new_file_path = file_path.replace('.csv', '_format.xlsx')

    total_rows = sum(1 for line in open(file_path)) - 1  # Subtract header row
    print(f'Total rows in CSV: {total_rows}')

    for chunk_number, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        print(f'Processing chunk {chunk_number + 1}...')

        processed_chunk = process_chunk(chunk)
        total_processed += len(chunk)
        total_valid += len(processed_chunk)

        if chunk_number == 0:
            processed_chunk.to_excel(new_file_path, index=False)
        else:
            append_df_to_excel(new_file_path, processed_chunk, index=False, header=False)

        print(f'Processed {total_processed} rows, Valid entries so far: {total_valid}')

    print(f'Total processed entries: {total_processed}')
    print(f'Total valid entries: {total_valid}')
    print(f'Entries dropped: {total_processed - total_valid}')
    print(f'Results saved to {new_file_path}')


if __name__ == "__main__":
    main()