import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor


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


def process_file(file_path, aggregated_metrics):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        initial_sim_result = flatten_result(data['initial_data']['sim_result'])
        sim_results = [flatten_result(step['sim_result']) for step in data['steps_data']]

        update_aggregated_metrics(aggregated_metrics, initial_sim_result, sim_results)
    except EOFError:
        print(f"Warning: Unable to read {file_path}, skipping.")


def scan_and_aggregate_metrics(input_folder, output_file):
    """遍历文件夹并汇总metrics数据"""
    aggregated_metrics = {}
    files = [os.path.join(root, file) for root, dirs, files in os.walk(input_folder) for file in files if file.endswith('.pkl')]
    total_files = len(files)
    print(f"Found {total_files} pickle files in the folder.")

    with ThreadPoolExecutor() as executor:
        for i, file_path in enumerate(files, 1):
            executor.submit(process_file, file_path, aggregated_metrics)
            print(f"Processing file {i}/{total_files}: {file_path}")

    # 保存整合后的数据为pickle文件
    with open(output_file, 'wb') as f:
        pickle.dump(aggregated_metrics, f)
    print("Aggregation complete. Output saved.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_file>")
    else:
        input_folder = sys.argv[1]
        output_file = sys.argv[2]
        scan_and_aggregate_metrics(input_folder, output_file)