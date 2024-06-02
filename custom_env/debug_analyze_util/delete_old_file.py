import argparse
import os
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


# Step 1: Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Delete old files and folders.")
    parser.add_argument("directory", type=str, help="Path to the directory to clean up")
    return parser.parse_args()


# Function to delete files
def delete_file(file_path):
    try:
        os.remove(file_path)
        return f"Deleted file: {file_path}"
    except Exception as e:
        return f"Failed to delete file: {file_path}, Error: {str(e)}"


# Function to delete directories
def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
        return f"Deleted directory: {dir_path}"
    except Exception as e:
        return f"Failed to delete directory: {dir_path}, Error: {str(e)}"


# Step 2: Define the main function
def main():
    args = parse_args()
    target_dir = args.directory
    current_time = time.time()
    cutoff_time = current_time - 24 * 3600  # 24 hours ago

    file_tasks = []
    dir_tasks = []

    # Step 3: Scan the directory
    for root, dirs, files in os.walk(target_dir):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.getmtime(file_path) < cutoff_time:
                file_tasks.append(file_path)

        for name in dirs:
            dir_path = os.path.join(root, name)
            if os.path.getmtime(dir_path) < cutoff_time:
                dir_tasks.append(dir_path)

    total_tasks = len(file_tasks) + len(dir_tasks)
    completed_tasks = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {executor.submit(delete_file, file): file for file in file_tasks}
        future_to_path.update({executor.submit(delete_directory, dir): dir for dir in dir_tasks})

        for future in as_completed(future_to_path):
            result = future.result()
            completed_tasks += 1
            print(f"Task {completed_tasks}/{total_tasks}: {result}")


if __name__ == "__main__":
    main()