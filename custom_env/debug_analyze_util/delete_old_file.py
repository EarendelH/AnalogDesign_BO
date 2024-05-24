import os
import shutil
import time
import concurrent.futures


def delete_old_file_or_dir(path):
    current_time = time.time()
    if os.path.isfile(path):
        file_time = os.path.getmtime(path)
        if current_time - file_time > 12 * 60 * 60:
            os.remove(path)
            print(f"Deleted file: {path}")
    elif os.path.isdir(path):
        dir_time = os.path.getmtime(path)
        if current_time - dir_time > 12 * 60 * 60:
            shutil.rmtree(path)
            print(f"Deleted directory: {path}")


def delete_old_files_and_dirs(base_path):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for root_dir, dirs, files in os.walk(base_path, topdown=False):
            executor.map(delete_old_file_or_dir, [os.path.join(root_dir, name) for name in files + dirs])


delete_old_files_and_dirs(input("Please input the path: "))
