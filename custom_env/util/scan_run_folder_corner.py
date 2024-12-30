import os
import pickle
import csv
from multiprocessing import Pool
import yaml
import importlib
from functools import partial
from tqdm import tqdm
import numpy as np


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types.
    Handle both numpy arrays and native Python types.

    Args:
        obj: Any Python object that might contain numpy values

    Returns:
        Converted object with numpy types replaced by native Python types
    """
    # Return directly if obj is None
    if obj is None:
        return None

    # Handle numpy scalars (including all numpy numeric types)
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
        return [convert_numpy_types(item) for item in obj]

    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)

    # Return directly if obj is already a Python native type
    if isinstance(obj, (int, float, str, bool)):
        return obj

    return obj


def parse_parameters(file_path):
    """
    Parse parameters from SCS file.

    Args:
        file_path: Path to the SCS file

    Returns:
        Dictionary of parameters
    """
    parameters_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('parameters'):
                parameters_line = line.strip().split(' ', 1)[1]
                parameters_pairs = parameters_line.split()
                for pair in parameters_pairs:
                    key, value = pair.split('=')
                    parameters_dict[key] = value
                break
    return parameters_dict


def process_folder(folder_path, cal_rew_func=None, ideal_specs_dict=None, norm_specs_dict=None):
    """
    Process a single folder containing simulation results.

    Args:
        folder_path: Path to the folder
        cal_rew_func: Optional reward calculation function
        ideal_specs_dict: Optional dictionary of ideal specifications
        norm_specs_dict: Optional dictionary of normalization specifications

    Returns:
        Tuple containing processed data or None if processing fails
    """
    parameters_dict = {}
    scs_files = [f for f in os.listdir(folder_path) if f.endswith('.scs')]
    scs_files.sort()
    if scs_files:
        scs_path = os.path.join(folder_path, scs_files[0])
        parameters_dict = parse_parameters(scs_path)

    folder_name = os.path.basename(folder_path)
    tag_parts = folder_name.split('_', 2)
    tag_name = tag_parts[2] if len(tag_parts) > 2 else None

    for folder_file in os.listdir(folder_path):
        if folder_file.endswith(".pkl"):
            with open(os.path.join(folder_path, folder_file), "rb") as f:
                step_data = pickle.load(f)
            try:
                sim_result = step_data['sim_result']
                if cal_rew_func:
                    # Convert specs before reward calculation
                    sim_result_converted = convert_numpy_types(sim_result)
                    rew = cal_rew_func(ideal_specs_dict, sim_result_converted, norm_specs_dict)
                else:
                    rew = convert_numpy_types(step_data.get('reward'))
                    sim_result = convert_numpy_types(sim_result)
            except Exception as e:
                print(f"Error occurred in processing reward: {str(e)}. Skipping and continuing.")
                rew = None
                sim_result = None
            return folder_name, tag_name, rew, sim_result, parameters_dict
    return None


def write_to_csv(results, output_csv_path):
    """
    Write processed results to CSV file.

    Args:
        results: List of processed results
        output_csv_path: Path for output CSV file
    """
    print("Writing results to CSV...")
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Folder Name", "Tag", "Reward", "Specs", "Parameters"])
        writer.writerows([result for result in results if result])


def main():
    """
    Main function to process folders and generate CSV output.
    """
    run_test_path = input("Enter the path of the run_test folder: ")
    output_csv_path = input("Enter the path of the output csv file: ")
    folder_paths = [os.path.join(run_test_path, folder) for folder in os.listdir(run_test_path)
                    if os.path.isdir(os.path.join(run_test_path, folder))]

    update_reward = input("Update reward? (y/n): ")
    if update_reward.lower() == 'y':
        config_path = input("Enter the path of the config folder: ")
        reward_func = input("Enter the reward function name: ")
        with open(os.path.join(config_path, "norm_specs.yaml"), 'r') as file:
            norm_specs = yaml.safe_load(file)
        with open(os.path.join(config_path, "norm_specs_cal.yaml"), 'r') as file:
            ideal_specs = yaml.safe_load(file)

        try:
            reward_module = importlib.import_module('cal_reward')
            cal_reward_func = getattr(reward_module, reward_func)
        except (ImportError, AttributeError) as e:
            print(f"Error importing reward function '{reward_func}': {e}")
            raise

        process_folder_partial = partial(process_folder, cal_rew_func=cal_reward_func,
                                         ideal_specs_dict=ideal_specs, norm_specs_dict=norm_specs)
    else:
        process_folder_partial = process_folder

    print("Processing folders...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_folder_partial, folder_paths),
                            total=len(folder_paths), desc="Processing folders"))

    write_to_csv(results, output_csv_path)
    print(f"Processing complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()