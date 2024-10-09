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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def flatten_nested_dict(d):
    flattened_dict = {}
    for key, nested_dict in d.items():
        flattened_dict.update(nested_dict)
    return flattened_dict


def is_valid_entry(flattened_dict):
    return all(value not in [0.0, 100.0] for value in flattened_dict.values())


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


def plot_label_distribution(labels, group_size=1000, output_path='valid_spec_count_distribution.png'):
    label_values = list(labels.values())
    max_label = max(label_values) if label_values else 0
    colors = LinearSegmentedColormap.from_list("gradient", ["#FFFFFF", "#1C4E87"], N=max_label + 1)

    num_groups = (len(label_values) + group_size - 1) // group_size
    results = []

    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, len(label_values))
        current_group = label_values[start:end]
        counts = {label: 0 for label in range(max_label + 1)}
        for label in current_group:
            counts[label] = counts.get(label, 0) + 1
        proportions = {key: value / len(current_group) * 100 for key, value in counts.items()}
        results.append(proportions)

    fig, ax = plt.subplots(figsize=(15, 10))
    base = np.zeros(num_groups)
    for label in range(max_label + 1):
        heights = [result.get(label, 0) for result in results]
        ax.bar(np.arange(num_groups) * group_size, heights, bottom=base, width=group_size, label=f'Count {label}',
               color=colors(label))
        base = np.add(base, heights)

    ax.set_xlabel('Directory Number Range')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Distribution of Valid Spec Counts per Group of Directories')
    ax.set_xticks(np.arange(0, group_size * num_groups, group_size))
    ax.set_xticklabels([f"{x}-{x + group_size - 1}" for x in range(0, len(label_values), group_size)], rotation=45)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Valid spec count distribution plot saved as '{output_path}'")


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

    print("Calculating valid spec count...")
    expanded_df['ValidSpecCount'] = expanded_df['Valid_Specs'].apply(
        lambda x: sum(1 for v in x.values() if v not in [0.0, 100.0]))
    valid_df = expanded_df[expanded_df['ValidSpecCount'] > 0]

    print("Saving to Parquet files...")
    pq.write_table(pa.Table.from_pandas(expanded_df), f"{output_parquet_prefix}_all_data.parquet")
    pq.write_table(pa.Table.from_pandas(valid_df), f"{output_parquet_prefix}_valid_data.parquet")

    print("Plotting valid spec count distribution...")
    labels = dict(zip(expanded_df['Folder Name'], expanded_df['ValidSpecCount']))
    plot_label_distribution(labels, output_path=f"{output_parquet_prefix}_valid_spec_count_distribution.png")

    print(
        f"Processing complete. Results saved to {output_parquet_prefix}_all_data.parquet and {output_parquet_prefix}_valid_data.parquet")
    print(f"Valid spec count distribution plot saved as '{output_parquet_prefix}_valid_spec_count_distribution.png'")


if __name__ == "__main__":
    main()