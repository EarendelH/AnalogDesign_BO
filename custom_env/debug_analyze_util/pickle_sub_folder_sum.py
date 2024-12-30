import os
import pickle
import yaml
import numpy as np


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types.

    Args:
        obj: Any Python object that might contain numpy values

    Returns:
        Converted object with numpy types replaced by native Python types
    """
    # Handle numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}

    # Handle lists
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in list]

    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in tuple)

    return obj


def pkl_to_dict(pkl_file):
    """
    Convert pickle file to dictionary with numpy types converted to native Python types.

    Args:
        pkl_file: Path to pickle file

    Returns:
        Dictionary with numpy types converted to native Python types
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return convert_numpy_types(data)


def dict_to_yaml(data_dict):
    """
    Convert dictionary to YAML string with improved formatting.

    Args:
        data_dict: Dictionary to convert

    Returns:
        YAML formatted string
    """
    return yaml.dump(data_dict,
                     default_flow_style=False,  # More readable block style
                     sort_keys=False,  # Preserve dictionary order
                     allow_unicode=True,  # Support unicode characters
                     width=1000)  # Avoid unnecessary line breaks


def scan_and_convert_pkl_files(source_dir):
    """
    Scan directory for pickle files and convert them to YAML.

    Args:
        source_dir: Directory containing pickle files
    """
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.pkl'):
                pkl_path = os.path.join(root, filename)
                print(f"Processing file: {pkl_path}")
                try:
                    data_dict = pkl_to_dict(pkl_path)
                    yaml_data = dict_to_yaml(data_dict)
                    yaml_file = os.path.join(root, os.path.splitext(filename)[0] + '.yaml')
                    with open(yaml_file, 'w') as f:
                        f.write(yaml_data)
                    print(f"Successfully converted to: {yaml_file}")
                except Exception as e:
                    print(f"Error processing {pkl_path}: {str(e)}")


if __name__ == '__main__':
    source_dir = input('Path to directory with pickle files: ')
    scan_and_convert_pkl_files(source_dir)