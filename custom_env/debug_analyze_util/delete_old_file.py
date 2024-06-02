import argparse
import os
import time
import threading

# Step 1: Get the folder path from arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Delete old files and folders.")
    parser.add_argument("folder_path", type=str, help="Path to the folder to scan and delete old files.")
    return parser.parse_args()

def scan_and_delete(folder_path):
    threads = []
    for root, dirs, files in os.walk(folder_path):
        for name in dirs + files:
            path = os.path.join(root, name)
            t = threading.Thread(target=check_and_delete, args=(path,))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

def check_and_delete(path):
    now = time.time()
    if os.path.exists(path):
        file_time = os.path.getmtime(path)
        if now - file_time > 24 * 3600:  # Older than 24 hours
            if os.path.isdir(path):
                try:
                    os.rmdir(path)
                    print(f"Deleted folder: {path}")
                except OSError:  # Directory not empty
                    for root, dirs, files in os.walk(path, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(path)
            else:
                os.remove(path)
                print(f"Deleted file: {path}")

if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder_path
    scan_and_delete(folder_path)