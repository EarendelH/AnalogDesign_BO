import argparse
import os
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Delete old files and folders.")
    parser.add_argument("folder_path", type=str, help="The path of the folder to scan and clean.")
    return parser.parse_args()


def get_all_files_and_folders(folder_path):
    items = []
    for root, dirs, files in os.walk(folder_path):
        for name in tqdm(dirs, desc="Scanning directories", unit="dir"):
            items.append(os.path.join(root, name))
        for name in tqdm(files, desc="Scanning files", unit="file"):
            items.append(os.path.join(root, name))
    return items


def is_older_than_24_hours(item_path):
    modification_time = os.path.getmtime(item_path)
    return (datetime.now() - datetime.fromtimestamp(modification_time)) > timedelta(hours=24)


def delete_item(item_path):
    try:
        if os.path.isdir(item_path):
            os.rmdir(item_path)
        else:
            os.remove(item_path)
    except Exception as e:
        print(f"Error deleting {item_path}: {e}")


def delete_old_items_concurrently(items):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(delete_item, item) for item in items]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f"Progress: {i+1}/{len(items)}")


def main():
    args = parse_args()
    folder_path = args.folder_path
    all_items = get_all_files_and_folders(folder_path)
    items_to_delete = [item for item in all_items if is_older_than_24_hours(item)]
    delete_old_items_concurrently(items_to_delete)


if __name__ == "__main__":
    main()
