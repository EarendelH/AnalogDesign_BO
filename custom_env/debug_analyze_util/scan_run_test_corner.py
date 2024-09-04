import os
import pickle
import csv
from multiprocessing import Pool


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


def process_folder(folder_path):
    parameters_dict = {}
    scs_files = [f for f in os.listdir(folder_path) if f.endswith('.scs')]
    scs_files.sort()
    if scs_files:
        scs_path = os.path.join(folder_path, scs_files[0])
        parameters_dict = parse_parameters(scs_path)

    for file in os.listdir(folder_path):
        if file.endswith(".pkl"):
            with open(os.path.join(folder_path, file), "rb") as f:
                step_data = pickle.load(f)
            try:
                rew = step_data['reward']
                rew = float(rew)
                corner_tag = step_data['corner']
                sim_result = step_data['sim_result']
            except Exception as e:
                print(f"Error occurred in cal_reward: {str(e)}. Skipping and continuing.")
                rew = None
                corner_tag = None
                sim_result = None
            return folder_path, rew, corner_tag, sim_result, parameters_dict
    return None


if __name__ == '__main__':
    run_test_path = input("Enter the path of the run_test folder: ")
    output_csv_path = input("Enter the path of the output csv file: ")

    folder_paths = [os.path.join(run_test_path, folder) for folder in os.listdir(run_test_path)
                    if os.path.isdir(os.path.join(run_test_path, folder))]

    with Pool() as pool:
        results = pool.map(process_folder, folder_paths)

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Folder Name", "Corner", "Reward", "Specs", "Parameters"])
        writer.writerows([result for result in results if result])