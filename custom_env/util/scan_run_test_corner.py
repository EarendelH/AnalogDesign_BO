import os
import pickle
import csv
from multiprocessing import Pool
import yaml
import importlib
from functools import partial


def parse_parameters(file_path):
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


def process_folder_update_rew(folder_path, cal_rew_func, ideal_specs_dict, norm_specs_dict):
    parameters_dict = {}
    scs_files = [f for f in os.listdir(folder_path) if f.endswith('.scs')]
    scs_files.sort()
    if scs_files:
        scs_path = os.path.join(folder_path, scs_files[0])
        parameters_dict = parse_parameters(scs_path)

    # Extract tag from folder name
    folder_name = os.path.basename(folder_path)
    tag_parts = folder_name.split('_', 2)
    tag_name = tag_parts[2] if len(tag_parts) > 2 else None

    for folder_file in os.listdir(folder_path):
        if folder_file.endswith(".pkl"):
            with open(os.path.join(folder_path, folder_file), "rb") as f:
                step_data = pickle.load(f)
            try:
                sim_result = step_data['sim_result']
                rew = cal_rew_func(ideal_specs_dict, sim_result, norm_specs_dict)
            except Exception as e:
                print(f"Error occurred in cal_reward: {str(e)}. Skipping and continuing.")
                rew = None
                sim_result = None
            return folder_name, rew, tag_name, sim_result, parameters_dict
    return None


def process_folder(folder_path):
    parameters_dict = {}
    scs_files = [f for f in os.listdir(folder_path) if f.endswith('.scs')]
    scs_files.sort()
    if scs_files:
        scs_path = os.path.join(folder_path, scs_files[0])
        parameters_dict = parse_parameters(scs_path)

    # Extract tag from folder name
    folder_name = os.path.basename(folder_path)
    tag_parts = folder_name.split('_', 2)
    tag_name = tag_parts[2] if len(tag_parts) > 2 else None

    for folder_file in os.listdir(folder_path):
        if folder_file.endswith(".pkl"):
            with open(os.path.join(folder_path, folder_file), "rb") as f:
                step_data = pickle.load(f)
            try:
                rew = step_data['reward']
                rew = float(rew)
                sim_result = step_data['sim_result']
            except Exception as e:
                print(f"Error occurred in cal_reward: {str(e)}. Skipping and continuing.")
                rew = None
                sim_result = None
            return folder_name, rew, tag_name, sim_result, parameters_dict
    return None


if __name__ == '__main__':
    run_test_path = input("Enter the path of the run_test folder: ")
    output_csv_path = input("Enter the path of the output csv file: ")
    folder_paths = [os.path.join(run_test_path, folder) for folder in os.listdir(run_test_path)
                    if os.path.isdir(os.path.join(run_test_path, folder))]
    # Whether update the reward in the csv file
    update_reward = input("Update reward? (y/n): ")
    if update_reward.lower() == 'y':
        config_path = input("Enter the path of the config folder: ")
        reward_func = input("Enter the reward function name: ")
        with open(os.path.join(config_path, "norm_specs.yaml"), 'r') as file:
            norm_specs = yaml.safe_load(file)
        with open(os.path.join(config_path, "norm_specs_cal.yaml"), 'r') as file:
            ideal_specs = yaml.safe_load(file)

        # Import the reward function
        try:
            reward_module = importlib.import_module('cal_reward')
            cal_reward_func = getattr(reward_module, reward_func)
        except (ImportError, AttributeError) as e:
            print(f"Error importing reward function '{reward_func}': {e}")
            raise

        # Create a partial function with the reward function with only folder update
        process_folder_update_rew_partial = partial(process_folder_update_rew, cal_rew_func=cal_reward_func,
                                     ideal_specs_dict=ideal_specs, norm_specs_dict=norm_specs)

        with Pool() as pool:
            results = pool.map(process_folder_update_rew_partial, folder_paths)

    elif update_reward.lower() == 'n':
        with Pool() as pool:
            results = pool.map(process_folder, folder_paths)
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        raise ValueError

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Folder Name", "Tag", "Reward", "Specs", "Parameters"])
        writer.writerows([result for result in results if result])