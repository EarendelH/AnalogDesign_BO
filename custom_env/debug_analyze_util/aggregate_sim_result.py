import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import dump


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
        init_data = {'result': initial_sim_result, 'param': init_param}
        # Convert to list
        sum_data = [init_data]

        # Iterate over the steps data, store the result and the updated param
        for step in data['steps_data']:
            step_sim_result = flatten_result(step['sim_result'])
            step_param = step['updated_param']
            step_data = {'result': step_sim_result, 'param': step_param}
            sum_data.append(step_data)

        # Print list size
        print(f"Processed {file_path} with {len(sum_data)} steps.")

        return sum_data
    except EOFError:
        print(f"Error reading file: {file_path}. File may be empty or corrupted.")
        return None, None


def scan_and_aggregate_metrics(input_folder, output_file):
    aggregated_metrics = []
    file_paths = [os.path.join(root, file)
                  for root, dirs, files in os.walk(input_folder)
                  for file in files if file.endswith('.pkl')]

    print(f"Found {len(file_paths)} pickle files.")

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            sum_data = future.result()
            if sum_data is not None:
                aggregated_metrics.extend(sum_data)
            print(f"Processed {i}/{len(file_paths)} files.")

    # Print all aggregated metrics size
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