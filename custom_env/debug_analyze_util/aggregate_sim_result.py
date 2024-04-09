import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import dump
from collections import OrderedDict


def flatten_result(sim_result):
    flat_result = {}
    for category, metrics in sim_result.items():
        for metric, value in metrics.items():
            flat_result[f"{metric}"] = value
    return flat_result


def process_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        initial_sim_result = flatten_result(data['initial_data']['sim_result'])
        init_param = data['initial_data']['init_param']
        # Convert to OrderedDict to maintain order of keys
        if isinstance(init_param, dict):
            init_param = OrderedDict(init_param.items())
        init_param = sorted(init_param.items())
        init_rew = data['initial_data']['rew']
        init_data = {'result': initial_sim_result, 'param': init_param, 'rew': init_rew}
        sum_data = [init_data]

        for step in data['steps_data']:
            step_sim_result = flatten_result(step['sim_result'])
            step_param = step['updated_param']
            # Convert to OrderedDict to maintain order of keys
            if isinstance(step_param, dict):
                step_param = OrderedDict(step_param.items())
            step_param = sorted(step_param.items())
            step_rew = step['rew']['Agent_1']
            step_data = {'result': step_sim_result, 'param': step_param, 'rew': step_rew}
            sum_data.append(step_data)

        print(f"Processed {file_path} with {len(sum_data)} steps.")
        return sum_data
    except (EOFError, pickle.UnpicklingError, Exception) as e:
        print(f"Error processing file: {file_path}. Skipping. Error: {e}")
        return None


def scan_and_aggregate_metrics(input_folder, output_file):
    aggregated_metrics = []
    file_info = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                last_modified_time = os.path.getmtime(file_path)
                file_info.append((file_path, last_modified_time))

    file_info.sort(key=lambda x: x[1])
    sorted_file_paths = [info[0] for info in file_info]

    print(f"Found {len(sorted_file_paths)} pickle files.")

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in sorted_file_paths}
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            sum_data = future.result()
            if sum_data is not None:
                aggregated_metrics.extend(sum_data)
            print(f"Processed {i}/{len(sorted_file_paths)} files.")

    print(f"Aggregated {len(aggregated_metrics)} metrics.")

    with open(output_file, 'wb') as f:
        dump(aggregated_metrics, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_file>")
    else:
        input_folder = sys.argv[1]
        output_file = sys.argv[2]
        scan_and_aggregate_metrics(input_folder, output_file)