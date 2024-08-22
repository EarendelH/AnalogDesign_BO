import csv
import yaml
import os


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


def csv2yaml(csv_file_path, yaml_file_path):
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]

    with open(yaml_file_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False, Dumper=NoQuotesDumper)


if __name__ == '__main__':
    csv_path = input("Enter the path of the csv file: ")
    yaml_path = os.path.splitext(csv_path)[0] + '.yaml'
    csv2yaml(csv_path, yaml_path)
    print(f"YAML file has been created at {yaml_path}")
