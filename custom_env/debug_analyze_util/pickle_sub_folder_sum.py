import os
import pickle
import yaml

def pkl_to_dict(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def dict_to_yaml(data_dict):
    return yaml.dump(data_dict)

def scan_and_convert_pkl_files(source_dir):
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.pkl'):
                pkl_path = os.path.join(root, filename)
                data_dict = pkl_to_dict(pkl_path)
                yaml_data = dict_to_yaml(data_dict)
                yaml_file = os.path.join(root, os.path.splitext(filename)[0] + '.yaml')
                with open(yaml_file, 'w') as f:
                    f.write(yaml_data)

if __name__ == '__main__':
    source_dir = input('Path to directory with pickle files: ')
    scan_and_convert_pkl_files(source_dir)
