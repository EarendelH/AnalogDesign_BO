import os
import shutil
import time

def delete_old_files_and_dirs(path):

    current_time = time.time()

    for root_dir, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root_dir, name)
            file_time = os.path.getmtime(file_path)
            if current_time - file_time > 12 * 60 * 60:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        for name in dirs:
            dir_path = os.path.join(root_dir, name)
            dir_time = os.path.getmtime(dir_path)
            if current_time - dir_time > 12 * 60 * 60:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")


delete_old_files_and_dirs(input("Please input the path: "))
