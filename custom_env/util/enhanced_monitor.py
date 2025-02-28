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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cleaning folder: {target_folder}")
    current_time = time.time()

    if not os.path.exists(target_folder):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: Target folder {target_folder} does not exist")
        return

    for entry in os.scandir(target_folder):
        if entry.is_dir():
            dir_path = entry.path
            dir_age = current_time - os.path.getmtime(dir_path)
            if dir_age > max_age:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found old directory: {dir_path} (age: {dir_age:.1f}s)")
                for sub_entry in os.scandir(dir_path):
                    sub_path = sub_entry.path
                    if sub_entry.is_dir():
                        try:
                            shutil.rmtree(sub_path)
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Deleted folder: {sub_path}")
                        except Exception as e:
                            print(
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to delete folder: {sub_path}. Error: {e}")
                    elif sub_entry.is_file():
                        if not (sub_entry.name.endswith('.scs') or sub_entry.name.endswith('.pkl')):
                            try:
                                os.remove(sub_path)
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Deleted file: {sub_path}")
                            except Exception as e:
                                print(
                                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to delete file: {sub_path}. Error: {e}")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Folder cleaning completed")


def check_directory_for_new_files(directory, time_window):
    """
    Check if new files have been created in the directory within the specified time window

    Args:
        directory: Directory to monitor
        time_window: Time window in seconds

    Returns:
        Boolean indicating if new files were detected
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for new files in: {directory}")
    current_time = time.time()

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directory {directory} does not exist.")
        return False

    # Get the most recent file modification time in the directory and its subdirectories
    most_recent_time = 0
    most_recent_file = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                mod_time = os.path.getmtime(file_path)
                if mod_time > most_recent_time:
                    most_recent_time = mod_time
                    most_recent_file = file_path
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error checking file {file_path}: {e}")

    # Check if the most recent file is within the time window
    if most_recent_time == 0:  # No files found
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No files found in {directory}")
        return False

    time_diff = current_time - most_recent_time

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Most recent file: {most_recent_file}")
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Last modified: {datetime.fromtimestamp(most_recent_time).strftime('%Y-%m-%d %H:%M:%S')} ({time_diff:.1f} seconds ago)")

    is_active = time_diff <= time_window

    if is_active:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directory is active (file modified within {time_window} seconds)")
    else:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directory is inactive (no modifications within {time_window} seconds)")

    return is_active


def find_latest_restore_folder(ray_results_dir):
    """
    Find the latest restore_* folder in the ray_results directory

    Args:
        ray_results_dir: Ray results directory path

    Returns:
        Tuple of (latest_restore_directory_path, restore_id)
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Searching for restore folders in: {ray_results_dir}")

    # Find all AnalogDesignEnv_v0 directories
    env_dirs = glob.glob(os.path.join(ray_results_dir, "AnalogDesignEnv*v0"))

    if not env_dirs:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No AnalogDesignEnv_v0 directories found in {ray_results_dir}")
        return None, None

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found environment directories: {env_dirs}")

    # Find all restore_* directories in all env directories
    restore_dirs = []
    for env_dir in env_dirs:
        pattern = os.path.join(env_dir, "restore_*")
        found_dirs = glob.glob(pattern)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Searching for pattern: {pattern}")
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(found_dirs)} restore directories in {env_dir}")
        restore_dirs.extend(found_dirs)

    if not restore_dirs:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No restore directories found")
        return None, None

    # Print all found restore directories with their modification times
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All restore directories found:")
    for dir_path in restore_dirs:
        mod_time = os.path.getmtime(dir_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {dir_path} (modified: {mod_time_str})")

    # Sort by modification time, newest first
    restore_dirs.sort(key=os.path.getmtime, reverse=True)

    latest_restore_dir = restore_dirs[0]
    restore_id = os.path.basename(latest_restore_dir)

    mod_time = os.path.getmtime(latest_restore_dir)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Latest restore directory: {latest_restore_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Restore ID: {restore_id}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Last modified: {mod_time_str}")

    return latest_restore_dir, restore_id


def find_latest_checkpoint(restore_dir):
    """
    Find the latest checkpoint_* directory in the given restore directory

    Args:
        restore_dir: Path to restore directory

    Returns:
        Path to latest checkpoint directory
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Searching for checkpoints in: {restore_dir}")

    checkpoint_dir = os.path.join(restore_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint directory {checkpoint_dir} does not exist.")
        return None

    # Find all checkpoint_* directories
    pattern = os.path.join(checkpoint_dir, "checkpoint_*")
    checkpoint_dirs = glob.glob(pattern)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Searching for pattern: {pattern}")

    if not checkpoint_dirs:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No checkpoint directories found")
        return None

    # Print all found checkpoint directories with their modification times
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All checkpoint directories found:")
    for dir_path in checkpoint_dirs:
        mod_time = os.path.getmtime(dir_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {dir_path} (modified: {mod_time_str})")

    # Sort by modification time, newest first
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)

    latest_checkpoint_dir = checkpoint_dirs[0]

    mod_time = os.path.getmtime(latest_checkpoint_dir)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Latest checkpoint directory: {latest_checkpoint_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Last modified: {mod_time_str}")

    return latest_checkpoint_dir


def update_yaml_config(config_path, checkpoint_path):
    """
    Update the checkpoint_path in the yaml config file

    Args:
        config_path: Path to config file
        checkpoint_path: New checkpoint path to set
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating checkpoint path in config: {config_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New checkpoint path: {checkpoint_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            old_path = config.get('checkpoint_path', 'None')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current checkpoint path: {old_path}")

        config['checkpoint_path'] = checkpoint_path

        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully updated checkpoint path")

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error updating YAML config: {e}")


def run_train_script(config_file, train_script_path):
    """
    Run the training script with the given config file using nohup

    Args:
        config_file: Path to config file
        train_script_path: Path to training script

    Returns:
        Tuple of (process_id, log_file)
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training process")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training script: {train_script_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Config file: {config_file}")

    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cmd = f"nohup python {train_script_path} --config_mode file --config_file {config_file} > {log_file} 2>&1 &"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running command: {cmd}")

    subprocess.run(cmd, shell=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command executed, waiting for process to start...")
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
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found matching process: PID={pid}, CMD={' '.join(cmdline)}")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error accessing process: {e}")

    if pid:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started training process with PID: {pid}, log file: {log_file}")
    else:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Started training process, but could not determine PID. Log file: {log_file}")

    return pid, log_file


def kill_process_tree(pid):
    """
    Kill the process with the given PID and all its child processes

    Args:
        pid: Process ID to kill

    Returns:
        Boolean indicating if kill was successful
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempting to kill process tree with root PID: {pid}")

    if pid is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No PID provided, cannot kill process.")
        return False

    try:
        parent = psutil.Process(pid)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found process: {parent.name()} (PID: {pid})")

        children = parent.children(recursive=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(children)} child processes")

        for i, child in enumerate(children):
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Child process {i + 1}: {child.name()} (PID: {child.pid})")
            child.terminate()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sent SIGTERM to child PID: {child.pid}")

        parent.terminate()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sent SIGTERM to parent PID: {pid}")

        # Wait for processes to terminate
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for processes to terminate...")
        gone, still_alive = psutil.wait_procs(children + [parent], timeout=5)

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {len(gone)} processes terminated, {len(still_alive)} still alive")

        # Force kill if still alive
        if still_alive:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Some processes still alive, sending SIGKILL...")
            for process in still_alive:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sending SIGKILL to PID: {process.pid}")
                process.kill()

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully terminated process tree with root PID: {pid}")
        return True
    except psutil.NoSuchProcess:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Process with PID {pid} not found")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error killing process with PID {pid}: {e}")
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing run folder")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Run test path: {run_test_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Restore ID: {restore_id}")

    # Create output directory
    output_dir = f"{output_base_dir}/{restore_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created output directory: {output_dir}")

    output_csv = f"{output_dir}/output_AXS.csv"

    # Run scan_run_folder_corner.py
    scan_cmd = f"python {scan_script_path} --run-path {run_test_path} --output-path {output_csv} --update-reward --config-path {config_path} --reward-func {reward_func}"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running scan_run_folder_corner.py")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command: {scan_cmd}")

    try:
        subprocess.run(scan_cmd, shell=True, check=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] scan_run_folder_corner.py completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error running scan_run_folder_corner.py: {e}")
        return output_dir

    # Run filitered_csv_format.py
    filter_cmd = f"python {filter_script_path} --input {output_csv} --filter"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running filitered_csv_format.py")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command: {filter_cmd}")

    try:
        subprocess.run(filter_cmd, shell=True, check=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] filitered_csv_format.py completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error running filitered_csv_format.py: {e}")

    return output_dir


def ensure_directory_exists(directory):
    """
    Ensure a directory exists, creating it if necessary

    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created directory: {directory}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directory already exists: {directory}")


def clean_run_test_directory(run_test_dir, to_be_deleted_dir, blank_dir):
    """
    Clean the run_test directory by moving it to to_be_deleted,
    using rsync to empty it, and then recreating an empty run_test directory

    Args:
        run_test_dir: Path to the run_test directory
        to_be_deleted_dir: Path to the to_be_deleted directory
        blank_dir: Path to a blank directory used for rsync
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cleaning run_test directory")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")

    # Ensure blank directory exists
    if not os.path.exists(blank_dir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating blank directory: {blank_dir}")
        os.makedirs(blank_dir, exist_ok=True)

    # Ensure to_be_deleted directory exists
    if not os.path.exists(to_be_deleted_dir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating to_be_deleted directory: {to_be_deleted_dir}")
        os.makedirs(to_be_deleted_dir, exist_ok=True)

    # Move run_test to to_be_deleted
    if os.path.exists(run_test_dir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Moving {run_test_dir} to {to_be_deleted_dir}")
        try:
            mv_cmd = f"mv {run_test_dir} {to_be_deleted_dir}"
            subprocess.run(mv_cmd, shell=True, check=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully moved directory")
        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error moving directory: {e}")
    else:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Run test directory {run_test_dir} does not exist, nothing to move")

    # Use rsync to clean to_be_deleted directory
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using rsync to clean {to_be_deleted_dir}")
    try:
        rsync_cmd = f"rsync --delete-before -a -H -v --progress --stats {blank_dir}/ {to_be_deleted_dir}"
        subprocess.run(rsync_cmd, shell=True, check=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully cleaned directory using rsync")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error running rsync: {e}")

    # Recreate run_test directory
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recreating {run_test_dir}")
    try:
        os.makedirs(run_test_dir, exist_ok=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully recreated run_test directory")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error creating directory: {e}")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Run test directory cleaning completed")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")


def safe_copy_directory(src, dst):
    """
    Safely copy a directory, handling the case where the destination already exists

    Args:
        src: Source directory
        dst: Destination directory
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Copying directory")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Source: {src}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Destination: {dst}")

    # Check Python version for dirs_exist_ok parameter
    try:
        if sys.version_info >= (3, 8):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using Python 3.8+ copytree with dirs_exist_ok")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            # For Python < 3.8, implement an alternative
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using Python <3.8 copytree alternative")
            if os.path.exists(dst):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Destination exists, removing it first")
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directory copied successfully")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error copying directory: {e}")


def main(monitor_dir, config_file, train_script_path, ray_results_dir, data_share_dir,
         scan_script_path, filter_script_path, config_dir, reward_func,
         inactivity_timeout, check_interval, debug_mode, to_be_deleted_dir, blank_dir):
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
        to_be_deleted_dir: Directory where run_test will be moved for deletion
        blank_dir: Blank directory used for rsync cleaning
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Enhanced Monitoring Script")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitoring directory: {monitor_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Configuration file: {config_file}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ray results directory: {ray_results_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data share directory: {data_share_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inactivity timeout: {inactivity_timeout} seconds")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Check interval: {check_interval} seconds")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Debug mode: {debug_mode}")

    # Make sure required directories exist
    checkpoint_dir = f"{data_share_dir}/checkpoint"
    run_dir = f"{data_share_dir}/run"

    ensure_directory_exists(checkpoint_dir)
    ensure_directory_exists(run_dir)
    ensure_directory_exists(monitor_dir)

    if debug_mode:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running in DEBUG mode - executing steps 4-8 only")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")

        # Step 4-5: Find latest restore folder and copy to data share
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 4-5: Finding latest restore folder and copying to data share")
        latest_restore_dir, restore_id = find_latest_restore_folder(ray_results_dir)
        if latest_restore_dir and restore_id:
            checkpoint_share_dir = f"{checkpoint_dir}/{restore_id}"
            ensure_directory_exists(os.path.dirname(checkpoint_share_dir))

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Copying {latest_restore_dir} to {checkpoint_share_dir}")
            safe_copy_directory(latest_restore_dir, checkpoint_share_dir)

            # Step 6-7: Process run folder
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 6-7: Processing run folder")
            process_run_folder(monitor_dir, restore_id, run_dir, scan_script_path, filter_script_path, config_dir,
                               reward_func)

            # Step 8: Update config file with latest checkpoint
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 8: Updating config file with latest checkpoint")
            latest_checkpoint = find_latest_checkpoint(latest_restore_dir)
            if latest_checkpoint:
                update_yaml_config(config_file, latest_checkpoint)

                # NEW STEP: Clean run_test directory in debug mode too
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 8.5: Cleaning run_test directory")
                clean_run_test_directory(monitor_dir, to_be_deleted_dir, blank_dir)
            else:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No checkpoint found, skipping config update and cleaning")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No restore folder found, skipping steps 5-8.5")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Debug run completed")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
        return

    # Main execution loop
    training_pid = None
    log_file = None

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting main execution loop")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")

    try:
        loop_counter = 0

        while True:
            loop_counter += 1
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loop iteration: {loop_counter}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")

            if training_pid is None:
                # Step 1: Start the training process
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1: Starting the training process")
                training_pid, log_file = run_train_script(config_file, train_script_path)
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started training process with PID {training_pid}, log file: {log_file}")
                time.sleep(check_interval)  # Give some time for the process to start

            # Step 2: Monitor and clean directory
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 2: Monitoring and cleaning directory")
            clean_folder(monitor_dir)

            # Step 3: Check if directory has new files in the last 30 minutes
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 3: Checking for new files in the last {inactivity_timeout} seconds")
            if not check_directory_for_new_files(monitor_dir, inactivity_timeout):
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No new files detected. Terminating training process.")
                if training_pid:
                    kill_process_tree(training_pid)
                    training_pid = None

                # Step 4-5: Find latest restore folder and copy to data share
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 4-5: Finding latest restore folder and copying to data share")
                latest_restore_dir, restore_id = find_latest_restore_folder(ray_results_dir)
                if latest_restore_dir and restore_id:
                    checkpoint_share_dir = f"{checkpoint_dir}/{restore_id}"
                    ensure_directory_exists(os.path.dirname(checkpoint_share_dir))

                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Copying {latest_restore_dir} to {checkpoint_share_dir}")
                    safe_copy_directory(latest_restore_dir, checkpoint_share_dir)

                    # Step 6-7: Process run folder
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 6-7: Processing run folder")
                    process_run_folder(monitor_dir, restore_id, run_dir, scan_script_path, filter_script_path,
                                       config_dir, reward_func)

                    # Step 8: Update config file with latest checkpoint
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 8: Updating config file with latest checkpoint")
                    latest_checkpoint = find_latest_checkpoint(latest_restore_dir)
                    if latest_checkpoint:
                        update_yaml_config(config_file, latest_checkpoint)

                        # NEW STEP: Clean run_test directory before restarting training
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 8.5: Cleaning run_test directory before restarting")
                        clean_run_test_directory(monitor_dir, to_be_deleted_dir, blank_dir)

                        # Step 9: Start training again
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 9: Starting training again")
                        training_pid, log_file = run_train_script(config_file, train_script_path)
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Restarted training process with PID {training_pid}, log file: {log_file}")
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No checkpoint found, skipping restart")
                else:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No restore folder found, skipping steps 5-9")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Directory is active, continuing monitoring")

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sleeping for {check_interval} seconds")
            time.sleep(check_interval)  # Check at the specified interval

    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script stopped by user")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========================================")
        if training_pid:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Terminating training process")
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

    # New parameters for run_test directory cleaning
    TO_BE_DELETED_DIR = "/home/wuhan/to_be_deleted"
    BLANK_DIR = "/home/wuhan/blank"

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
        debug_mode=args.debug,
        to_be_deleted_dir=TO_BE_DELETED_DIR,
        blank_dir=BLANK_DIR
    )