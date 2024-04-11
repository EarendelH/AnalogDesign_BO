import os
import pickle
import yaml
import csv


def cal_reward(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    :param ideal_specs_dict: Dict with ideal specs value and property
    :param cur_specs_dict: Dict with current specs
    :param norm_specs_dict: Dict with normalized specs
    :return: reward: float, reward value
    """

    # Flatten cur_specs_dict
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}
    norm_specs_flatten = {k: v for d in norm_specs_dict.values() for k, v in d.items()}
    rew = 0

    # epsilon = 0.1

    for spec, detail in ideal_specs_dict.items():

        single_reward = 0

        ideal_spec_value = float(detail['value'])
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_objective = detail['objective']

        if constrain_objective == "max":
            single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
        elif constrain_objective == "min":
            single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)

        rew += float(single_reward)

    if rew >= 0:
        rew = rew + 15
        for spec, detail in ideal_specs_dict.items():

            single_reward = 0
            general_ideal_spec_value = float(norm_specs_flatten[spec])
            cur_spec_value = float(cur_specs_flatten[spec])
            reward_type = detail['reward_type']
            constrain_objective = detail['objective']

            if reward_type == "optimal":
                if constrain_objective == "max":
                    single_reward = max((cur_spec_value - general_ideal_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
                elif constrain_objective == "min":
                    single_reward = max((general_ideal_spec_value - cur_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)

            rew += float(single_reward)*2  # 2 is the weight for the optimal reward

    return rew


run_test_path = input("Enter the path of the run_test folder: ")
config_path = input("Enter the path of the config folder: ")
output_csv_path = input("Enter the path of the output csv file: ")

norm_specs_file = config_path + "/norm_specs.yaml"

with open(norm_specs_file, 'r') as file:
    norm_specs = yaml.safe_load(file)

sim_summary = {}

# Iterate over all folder in the path
for folder in os.listdir(run_test_path):
    # Check if the folder is a directory
    if os.path.isdir(os.path.join(run_test_path, folder)):
        # Iterate over all files in the folder
        for file in os.listdir(os.path.join(run_test_path, folder)):
            # Check if the file is a pickle file
            if file.endswith(".pkl"):
                # Load the pickle file
                with open(os.path.join(run_test_path, folder, file), "rb") as f:
                    single_sim = pickle.load(f)
                # Create sub-dict for the folder
                sim_summary[folder] = {}
                sim_summary[folder]["result"] = single_sim
                print(f"Debug, norm_specs: {norm_specs}, single_sim: {single_sim}")
                rew_single = cal_reward(norm_specs, single_sim, norm_specs)
                sim_summary[folder]["reward"] = rew_single
                print(f"Reward for {folder} is {rew_single} with specs {single_sim}")

# Save the summary to a csv file
with open(output_csv_path, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Folder Name", "Reward", "Specs"])
    for key, value in sim_summary.items():
        writer.writerow([key, value["reward"], value["result"]])
