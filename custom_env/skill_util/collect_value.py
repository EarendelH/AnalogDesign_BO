import os
import argparse
import pandas as pd
import yaml
import openpyxl


class NoQuotesDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        if tag == 'tag:yaml.org,2002:str':
            try:
                int(value)
                return self.represent_scalar('tag:yaml.org,2002:int', value)
            except ValueError:
                try:
                    float(value)
                    return self.represent_scalar('tag:yaml.org,2002:float', value)
                except ValueError:
                    pass
        return super().represent_scalar(tag, value, style)


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def read_excel_file(file_path):
    workbook = openpyxl.load_workbook(file_path, read_only=True)
    sheet = workbook.active

    # Convert Excel data to pandas DataFrame
    data = sheet.values
    columns = next(data)[0:]  # Assumes the first row has the column names
    df = pd.DataFrame(data, columns=columns)

    return df


def extract_data(excel_df, yaml_data):
    extracted_data = []

    for index, row in excel_df.iterrows():
        data = {
            'index': index + 1,
            'id': row['Folder Name'][-6:],
            'Core_Param': {}
        }

        # 保留Core_Cellviews字段（如果存在）
        if 'Core_Cellviews' in yaml_data:
            data['Core_Cellviews'] = yaml_data['Core_Cellviews']

        # Extract Core_Param from Excel
        for param in yaml_data['Core_Param']:
            if param in row:
                data['Core_Param'][param] = row[param]

        # If Testbench_Param exists, extract it
        if 'Testbench_Param' in yaml_data:
            data['Testbench_Param'] = {}
            for param in yaml_data['Testbench_Param']:
                if param in row:
                    data['Testbench_Param'][param] = row[param]

        extracted_data.append(data)

    return extracted_data


def save_yaml_files(output_dir, data):
    value_dir = os.path.join(output_dir, 'value')
    os.makedirs(value_dir, exist_ok=True)

    for item in data:
        file_path = os.path.join(value_dir, f"{item['index']}.yaml")
        with open(file_path, 'w') as file:
            yaml.dump(item, file, default_flow_style=False, sort_keys=False, Dumper=NoQuotesDumper)


def collect_value(input_dir):
    # Read YAML File
    yaml_file = os.path.join(input_dir, 'info.yaml')
    yaml_data = read_yaml_file(yaml_file)

    # Read Excel File
    excel_file = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')][0]
    excel_path = os.path.join(input_dir, excel_file)
    excel_df = read_excel_file(excel_path)

    # Extract Data
    extracted_data = extract_data(excel_df, yaml_data)

    # Save YAML Files
    save_yaml_files(input_dir, extracted_data)