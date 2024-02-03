import os
import pickle
import sys


def flatten_result(sim_result):
    flat_result = {}
    for category, metrics in sim_result.items():
        for metric, value in metrics.items():
            flat_result[f"{metric}"] = value
    return flat_result


def update_aggregated_metrics(aggregated_metrics, initial_sim_result, sim_results):
    for metric, value in initial_sim_result.items():
        if metric not in aggregated_metrics:
            aggregated_metrics[metric] = [value]
        else:
            aggregated_metrics[metric].append(value)

    for sim_result in sim_results:
        for metric, value in sim_result.items():
            if metric not in aggregated_metrics:
                aggregated_metrics[metric] = [value]
            else:
                aggregated_metrics[metric].append(value)


def scan_and_aggregate_metrics(input_folder, output_file):
    aggregated_metrics = {}
    pickle_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    total_files = len(pickle_files)
    print(f"Found {total_files} pickle files in the folder.")

    for i, file in enumerate(pickle_files, start=1):
        print(f"Processing file {i} of {total_files}...")
        file_path = os.path.join(input_folder, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        initial_sim_result = flatten_result(data['initial_data']['sim_result'])
        sim_results = [flatten_result(step['sim_result']) for step in data['steps_data']]

        update_aggregated_metrics(aggregated_metrics, initial_sim_result, sim_results)

    with open(output_file, 'wb') as f:
        pickle.dump(aggregated_metrics, f)
    print(f"All files processed. Output saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_file>")
    else:
        input_folder = sys.argv[1]
        output_file = sys.argv[2]
        scan_and_aggregate_metrics(input_folder, output_file)