import os
import time
import shutil


def clean_folder(target_folder):
    current_time = time.time()

    for entry in os.scandir(target_folder):
        if entry.is_dir() and  entry.name.startswith('tmp') :
            dir_path = entry.path
            if current_time - os.path.getmtime(dir_path) > 300:
                for sub_entry in os.scandir(dir_path):
                    sub_path = sub_entry.path
                    if sub_entry.is_dir():
                        try:
                            shutil.rmtree(sub_path)
                            print(f"Deleted folder: {sub_path} "
                                  f"at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                        except Exception as e:
                            print(f"Failed to delete folder: {sub_path} "
                                  f"at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}. "
                                  f"Error: {e}")
                    elif sub_entry.is_file():
                        if not (sub_entry.name.endswith('.scs') or sub_entry.name.endswith('.pkl')):
                            try:
                                os.remove(sub_path)
                                print(f"Deleted file: {sub_path} "
                                      f"at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                            except Exception as e:
                                print(f"Failed to delete file: {sub_path} "
                                      f"at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}. "
                                      f"Error: {e}")


def main_loop(target_folder):
    try:
        while True:
            clean_folder(target_folder)
            time.sleep(60)
    except KeyboardInterrupt:
        print("Script stopped by user.")


# monitor_dir = input("Please input the directory you want to monitor: ")
#monitor_dir = "/home/wuhan/AnalogDesignAuto_BO/AnalogDesignAuto_MultiAgent/custom_env/run_AXS_BO_07181343/run_20250721_153107"
monitor_dir = "/home/wuhan/to_be_deleted/AnalogDesignAuto_BO/AnalogDesignAuto_MultiAgent/custom_env/run_bo_Mohamed_spectre/run_20260410_044501"
main_loop(monitor_dir)
