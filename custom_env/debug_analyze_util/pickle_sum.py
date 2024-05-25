import os
import pickle
import yaml

def pkl_to_dict(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def dict_to_yaml(data_dict):
    return yaml.dump(data_dict)

def scan_and_convert_pkl_files(source_dir, output_file):
    yaml_data = []
    for filename in os.listdir(source_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(source_dir, filename)
            data_dict = pkl_to_dict(pkl_path)
            yaml_data.append(dict_to_yaml(data_dict))
    with open(output_file, 'w') as f:
        f.write('\n---\n'.join(yaml_data))

if __name__ == '__main__':
    source_dir = input('Path to directory with pickle files: ')
    output_file = input('Path to output YAML file: ')
    scan_and_convert_pkl_files(source_dir, output_file)