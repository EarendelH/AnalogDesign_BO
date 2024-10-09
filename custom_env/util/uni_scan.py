import os
import pickle
import pandas as pd
import ast
from multiprocessing import Pool
import yaml
import importlib
from functools import partial
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


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


def process_folder(folder_path, cal_rew_func=None, ideal_specs_dict=None, norm_specs_dict=None):
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
                    rew = cal_rew_func(ideal_specs_dict, sim_result, norm_specs_dict)
                else:
                    rew = step_data.get('reward')
                rew = float(rew) if rew is not None else None
            except Exception as e:
                print(f"Error occurred in processing reward: {str(e)}. Skipping and continuing.")
                rew = None
                sim_result = None
            return folder_name, tag_name, rew, sim_result, parameters_dict
    return None


def expand_data(data):
    # Expand Specs
    data['Valid_Specs'] = data['Specs'].apply(lambda x: flatten_nested_dict(ast.literal_eval(x)))
    specs_df = data['Valid_Specs'].apply(pd.Series)

    # Expand Parameters
    params_df = data['Parameters'].apply(ast.literal_eval).apply(pd.Series)

    # Combine expanded data
    expanded_data = pd.concat([data.drop(columns=['Specs', 'Valid_Specs', 'Parameters']), specs_df, params_df], axis=1)

    return expanded_data


def main():
    run_test_path = input("Enter the path of the run_test folder: ")
    output_parquet_prefix = input("Enter the prefix for output Parquet files: ")

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

    print("Creating DataFrame...")
    df = pd.DataFrame(results, columns=["Folder Name", "Tag", "Reward", "Specs", "Parameters"])
    df = df.dropna()  # Remove any None results

    print("Expanding data...")
    expanded_df = expand_data(df)

    print("Filtering valid entries...")
    valid_df = expanded_df[expanded_df['Valid_Specs'].apply(is_valid_entry)]

    print("Saving to Parquet files...")
    pq.write_table(pa.Table.from_pandas(expanded_df), f"{output_parquet_prefix}_all_data.parquet")
    pq.write_table(pa.Table.from_pandas(valid_df), f"{output_parquet_prefix}_valid_data.parquet")

    print(
        f"Processing complete. Results saved to {output_parquet_prefix}_all_data.parquet and {output_parquet_prefix}_valid_data.parquet")


if __name__ == "__main__":
    main()