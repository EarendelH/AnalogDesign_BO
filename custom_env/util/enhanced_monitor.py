import os
import time
import shutil
import subprocess
import argparse
import glob
import yaml
import signal
import sys
import psutil
from datetime import datetime


def clean_folder(target_folder, max_age=300):
    """
    Clean folders that are older than max_age seconds

    Args:
        target_folder: Target directory to clean
        max_age: Maximum age of files in seconds (default: 300s)
    """
    current_time = time.time()

    for entry in os.scandir(target_folder):
        if entry.is_dir():
            dir_path = entry.path
            if current_time - os.path.getmtime(dir_path) > max_age:
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


def check_directory_for_new_files(directory, time_window):
    """
    Check if new files have been created in the directory within the specified time window

    Args:
        directory: Directory to monitor
        time_window: Time window in seconds

    Returns:
        Boolean indicating if new files were detected
    """
    current_time = time.time()

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return False

    # Get the most recent file modification time in the directory and its subdirectories
    most_recent_time = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            mod_time = os.path.getmtime(file_path)
            if mod_time > most_recent_time:
                most_recent_time = mod_time

    # Check if the most recent file is within the time window
    if most_recent_time == 0:  # No files found
        return False

    time_diff = current_time - most_recent_time

    print(f"Most recent file modification was {time_diff:.2f} seconds ago.")

    return time_diff <= time_window


def find_latest_restore_folder(ray_results_dir):
    """
    Find the latest restore_* folder in the ray_results directory

    Args:
        ray_results_dir: Ray results directory path

    Returns:
        Tuple of (latest_restore_directory_path, restore_id)
    """
    # Find all AnalogDesignEnv_v0 directories
    env_dirs = glob.glob(os.path.join(ray_results_dir, "AnalogDesignEnv*v0"))

    if not env_dirs:
        print("No AnalogDesignEnv_v0 directories found.")
        return None, None

    # Find all restore_* directories in all env directories
    restore_dirs = []
    for env_dir in env_dirs:
        restore_dirs.extend(glob.glob(os.path.join(env_dir, "restore_*")))

    if not restore_dirs:
        print("No restore_* directories found.")
        return None, None

    # Sort by modification time, newest first
    restore_dirs.sort(key=os.path.getmtime, reverse=True)

    latest_restore_dir = restore_dirs[0]
    restore_id = os.path.basename(latest_restore_dir)

    print(f"Found latest restore directory: {latest_restore_dir}")

    return latest_restore_dir, restore_id


def find_latest_checkpoint(restore_dir):
    """
    Find the latest checkpoint_* directory in the given restore directory

    Args:
        restore_dir: Path to restore directory

    Returns:
        Path to latest checkpoint directory
    """
    checkpoint_dir = os.path.join(restore_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None

    # Find all checkpoint_* directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))

    if not checkpoint_dirs:
        print("No checkpoint_* directories found.")
        return None

    # Sort by modification time, newest first
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)

    latest_checkpoint_dir = checkpoint_dirs[0]

    print(f"Found latest checkpoint directory: {latest_checkpoint_dir}")

    return latest_checkpoint_dir


def update_yaml_config(config_path, checkpoint_path):
    """
    Update the checkpoint_path in the yaml config file

    Args:
        config_path: Path to config file
        checkpoint_path: New checkpoint path to set
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        config['checkpoint_path'] = checkpoint_path

        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Updated checkpoint_path in {config_path} to {checkpoint_path}")

    except Exception as e:
        print(f"Error updating YAML config: {e}")


def run_train_script(config_file, train_script_path):
    """
    Run the training script with the given config file using nohup

    Args:
        config_file: Path to config file
        train_script_path: Path to training script

    Returns:
        Tuple of (process_id, log_file)
    """
    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cmd = f"nohup python {train_script_path} --config_mode file --config_file {config_file} > {log_file} 2>&1 &"

    subprocess.run(cmd, shell=True)

    # Wait a bit for the process to start
    time.sleep(5)

    # Find the process ID by looking for processes with this command line
    pid = None
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'train_continuous.py' in ' '.join(cmdline) and f'--config_file {config_file}' in ' '.join(
                    cmdline):
                pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if pid:
        print(f"Started training process with PID: {pid}, log file: {log_file}")
    else:
        print(f"Started training process, but could not determine PID. Log file: {log_file}")

    return pid, log_file


def kill_process_tree(pid):
    """
    Kill the process with the given PID and all its child processes

    Args:
        pid: Process ID to kill

    Returns:
        Boolean indicating if kill was successful
    """
    if pid is None:
        print("No PID provided, cannot kill process.")
        return False

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        for child in children:
            child.terminate()

        parent.terminate()

        # Wait for processes to terminate
        gone, still_alive = psutil.wait_procs(children + [parent], timeout=5)

        # Force kill if still alive
        for process in still_alive:
            process.kill()

        print(f"Terminated process tree with root PID: {pid}")
        return True
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found")
        return False
    except Exception as e:
        print(f"Error killing process with PID {pid}: {e}")
        return False


def process_run_folder(run_test_path, restore_id, output_base_dir, scan_script_path, filter_script_path, config_path,
                       reward_func):
    """
    Process the run_test folder using scan_run_folder_corner.py and filitered_csv_format.py

    Args:
        run_test_path: Path to run test directory
        restore_id: Restore ID
        output_base_dir: Base directory for output
        scan_script_path: Path to scan_run_folder_corner.py
        filter_script_path: Path to filitered_csv_format.py
        config_path: Path to config directory
        reward_func: Reward function name

    Returns:
        Path to output directory
    """
    # Create output directory
    output_dir = f"{output_base_dir}/{restore_id}"
    os.makedirs(output_dir, exist_ok=True)

    output_csv = f"{output_dir}/output_AXS.csv"

    # Run scan_run_folder_corner.py
    scan_cmd = f"python {scan_script_path} --run-path {run_test_path} --output-path {output_csv} --update-reward --config-path {config_path} --reward-func {reward_func}"

    print(f"Running scan_run_folder_corner.py: {scan_cmd}")
    subprocess.run(scan_cmd, shell=True, check=True)

    # Run filitered_csv_format.py
    filter_cmd = f"python {filter_script_path} --input {output_csv} --filter"

    print(f"Running filitered_csv_format.py: {filter_cmd}")
    subprocess.run(filter_cmd, shell=True, check=True)

    return output_dir


def ensure_directory_exists(directory):
    """
    Ensure a directory exists, creating it if necessary

    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def safe_copy_directory(src, dst):
    """
    Safely copy a directory, handling the case where the destination already exists

    Args:
        src: Source directory
        dst: Destination directory
    """
    # Check Python version for dirs_exist_ok parameter
    if sys.version_info >= (3, 8):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        # For Python < 3.8, implement an alternative
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main(monitor_dir, config_file, train_script_path, ray_results_dir, data_share_dir,
         scan_script_path, filter_script_path, config_dir, reward_func,
         inactivity_timeout, check_interval, debug_mode):
    """
    Main function implementing the automatic training monitor workflow

    Args:
        monitor_dir: Directory to monitor for cleanup
        config_file: Path to config file
        train_script_path: Path to training script
        ray_results_dir: Path to ray results directory
        data_share_dir: Path to data share directory
        scan_script_path: Path to scan_run_folder_corner.py
        filter_script_path: Path to filitered_csv_format.py
        config_dir: Path to config directory
        reward_func: Reward function name
        inactivity_timeout: Time in seconds to wait for new file activity
        check_interval: Time in seconds between checks
        debug_mode: Run only steps 4-8 for debugging
    """
    print(f"Starting Enhanced Monitoring Script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Monitoring directory: {monitor_dir}")
    print(f"Configuration file: {config_file}")

    # Make sure required directories exist
    checkpoint_dir = f"{data_share_dir}/checkpoint"
    run_dir = f"{data_share_dir}/run"

    ensure_directory_exists(checkpoint_dir)
    ensure_directory_exists(run_dir)
    ensure_directory_exists(monitor_dir)

    if debug_mode:
        print("Running in DEBUG mode - executing steps 4-8 only")

        # Step 4-5: Find latest restore folder and copy to data share
        latest_restore_dir, restore_id = find_latest_restore_folder(ray_results_dir)
        if latest_restore_dir and restore_id:
            checkpoint_share_dir = f"{checkpoint_dir}/{restore_id}"
            ensure_directory_exists(os.path.dirname(checkpoint_share_dir))

            print(f"Copying {latest_restore_dir} to {checkpoint_share_dir}")
            safe_copy_directory(latest_restore_dir, checkpoint_share_dir)

            # Step 6-7: Process run folder
            process_run_folder(monitor_dir, restore_id, run_dir, scan_script_path, filter_script_path, config_dir,
                               reward_func)

            # Step 8: Update config file with latest checkpoint
            latest_checkpoint = find_latest_checkpoint(latest_restore_dir)
            if latest_checkpoint:
                update_yaml_config(config_file, latest_checkpoint)

        print("Debug run completed")
        return

    # Main execution loop
    training_pid = None
    log_file = None

    try:
        while True:
            if training_pid is None:
                # Step 1: Start the training process
                training_pid, log_file = run_train_script(config_file, train_script_path)
                print(f"Started training process with PID {training_pid}, log file: {log_file}")
                time.sleep(check_interval)  # Give some time for the process to start

            # Step 2: Monitor and clean directory
            clean_folder(monitor_dir)

            # Step 3: Check if directory has new files in the last 30 minutes
            if not check_directory_for_new_files(monitor_dir, inactivity_timeout):
                print(f"No new files detected in the last {inactivity_timeout} seconds. Terminating training process.")
                if training_pid:
                    kill_process_tree(training_pid)
                    training_pid = None

                # Step 4-5: Find latest restore folder and copy to data share
                latest_restore_dir, restore_id = find_latest_restore_folder(ray_results_dir)
                if latest_restore_dir and restore_id:
                    checkpoint_share_dir = f"{checkpoint_dir}/{restore_id}"
                    ensure_directory_exists(os.path.dirname(checkpoint_share_dir))

                    print(f"Copying {latest_restore_dir} to {checkpoint_share_dir}")
                    safe_copy_directory(latest_restore_dir, checkpoint_share_dir)

                    # Step 6-7: Process run folder
                    process_run_folder(monitor_dir, restore_id, run_dir, scan_script_path, filter_script_path,
                                       config_dir, reward_func)

                    # Step 8: Update config file with latest checkpoint
                    latest_checkpoint = find_latest_checkpoint(latest_restore_dir)
                    if latest_checkpoint:
                        update_yaml_config(config_file, latest_checkpoint)

                    # Step 9: Start training again
                    training_pid, log_file = run_train_script(config_file, train_script_path)
                    print(f"Restarted training process with PID {training_pid}, log file: {log_file}")

            time.sleep(check_interval)  # Check at the specified interval

    except KeyboardInterrupt:
        print("Script stopped by user.")
        if training_pid:
            kill_process_tree(training_pid)
        sys.exit(0)


if __name__ == "__main__":
    # Hardcoded configuration parameters
    MONITOR_DIR = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/run_test"
    CONFIG_FILE = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/train_config/AXS_eex.yaml"
    CONFIG_FILE = os.path.expanduser(CONFIG_FILE)

    TRAIN_SCRIPT_PATH = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/train_continuous.py"
    RAY_RESULTS_DIR = "/home/wuhan/ray_results"
    DATA_SHARE_DIR = "/data/share/train_data/AXS"

    SCAN_SCRIPT_PATH = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/util/scan_run_folder_corner.py"
    FILTER_SCRIPT_PATH = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/debug_analyze_util/filitered_csv_format.py"
    CONFIG_DIR = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/config/config_AXS_Simple"
    REWARD_FUNC = "cal_reward_AXS"

    INACTIVITY_TIMEOUT = 1800  # 30 minutes in seconds
    CHECK_INTERVAL = 60  # 1 minute in seconds

    # Parse command-line arguments to allow overriding the hardcoded values
    parser = argparse.ArgumentParser(description="Enhanced monitoring script for Analog Design AutoRL")
    parser.add_argument("--monitor-dir", help="Directory to monitor for cleanup")
    parser.add_argument("--debug", action="store_true", help="Run only steps 4-8 for debugging")
    parser.add_argument("--config-file", help="Path to the YAML config file")
    parser.add_argument("--inactivity-timeout", type=int,
                        help="Time in seconds to wait for new file activity before restarting")
    parser.add_argument("--check-interval", type=int, help="Time in seconds between checks")

    args = parser.parse_args()

    # Override hardcoded values with command-line arguments if provided
    if args.monitor_dir:
        MONITOR_DIR = args.monitor_dir
    if args.config_file:
        CONFIG_FILE = os.path.expanduser(args.config_file)
    if args.inactivity_timeout:
        INACTIVITY_TIMEOUT = args.inactivity_timeout
    if args.check_interval:
        CHECK_INTERVAL = args.check_interval

    # Call the main function with all parameters
    main(
        monitor_dir=MONITOR_DIR,
        config_file=CONFIG_FILE,
        train_script_path=TRAIN_SCRIPT_PATH,
        ray_results_dir=RAY_RESULTS_DIR,
        data_share_dir=DATA_SHARE_DIR,
        scan_script_path=SCAN_SCRIPT_PATH,
        filter_script_path=FILTER_SCRIPT_PATH,
        config_dir=CONFIG_DIR,
        reward_func=REWARD_FUNC,
        inactivity_timeout=INACTIVITY_TIMEOUT,
        check_interval=CHECK_INTERVAL,
        debug_mode=args.debug
    )