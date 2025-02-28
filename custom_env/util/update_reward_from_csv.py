"""
Utility script to update rewards in an existing CSV file using a specified reward function.
This script reads a CSV file with simulation results, recalculates rewards based on a new
reward function, and writes the updated data to a new CSV file.
"""

import os
import argparse
import csv
import ast
import yaml
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    """
    Parse command line arguments

    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Update rewards in CSV file using new reward function')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file with simulation results')
    parser.add_argument('--output', type=str,
                        help='Path to output updated CSV file (default: input_updated.csv)')
    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to the config folder containing norm_specs.yaml and norm_specs_cal.yaml')
    parser.add_argument('--reward-func', type=str, required=True,
                        help='Name of reward function to use')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed output')

    args = parser.parse_args()

    # Set default output path if not provided
    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_updated.csv"

    return args


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types.
    Handle both numpy arrays and native Python types.

    Args:
        obj: Any Python object that might contain numpy values

    Returns:
        Converted object with numpy types replaced by native Python types
    """
    if obj is None:
        return None

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)

    if isinstance(obj, (int, float, str, bool)):
        return obj

    return obj


def load_configs(config_path):
    """
    Load configuration files

    Args:
        config_path: Path to the config folder

    Returns:
        Tuple of (norm_specs, ideal_specs)
    """
    norm_specs_path = os.path.join(config_path, "norm_specs.yaml")
    ideal_specs_path = os.path.join(config_path, "norm_specs_cal.yaml")

    if not os.path.exists(norm_specs_path):
        raise FileNotFoundError(f"norm_specs.yaml not found at {norm_specs_path}")

    if not os.path.exists(ideal_specs_path):
        raise FileNotFoundError(f"norm_specs_cal.yaml not found at {ideal_specs_path}")

    with open(norm_specs_path, 'r') as file:
        norm_specs = yaml.safe_load(file)

    with open(ideal_specs_path, 'r') as file:
        ideal_specs = yaml.safe_load(file)

    return norm_specs, ideal_specs


def import_reward_function(reward_func_name):
    """
    Import the specified reward function

    Args:
        reward_func_name: Name of the reward function

    Returns:
        Imported reward function
    """
    try:
        # First try to import from util.cal_reward
        reward_module = importlib.import_module('util.cal_reward')
        cal_reward_func = getattr(reward_module, reward_func_name)
        return cal_reward_func
    except (ImportError, AttributeError):
        try:
            # If that fails, try importing directly from cal_reward
            reward_module = importlib.import_module('cal_reward')
            cal_reward_func = getattr(reward_module, reward_func_name)
            return cal_reward_func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error importing reward function '{reward_func_name}': {e}")


def update_rewards(input_csv, output_csv, cal_reward_func, norm_specs, ideal_specs, debug=False):
    """
    Update rewards in a CSV file

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        cal_reward_func: Reward calculation function
        norm_specs: Normalization specifications
        ideal_specs: Ideal specifications
        debug: Boolean indicating if debug mode is enabled
    """
    if debug:
        print(f"Reading CSV file: {input_csv}")

    # Read the input CSV file
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"Processing {total_rows} rows from {input_csv}")

    # Create a new column for updated rewards
    df['Updated_Reward'] = None

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Updating rewards"):
        try:
            # Convert specs string to dictionary
            specs_str = row['Specs']
            specs_dict = ast.literal_eval(specs_str)

            # Convert numpy types to Python native types
            specs_dict = convert_numpy_types(specs_dict)

            # Calculate new reward
            new_reward = cal_reward_func(ideal_specs, specs_dict, norm_specs)

            # Update the reward
            df.at[idx, 'Updated_Reward'] = new_reward

            if debug and idx < 5:  # Show first 5 for debugging
                print(f"Row {idx}: Original reward = {row['Reward']}, Updated reward = {new_reward}")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            df.at[idx, 'Updated_Reward'] = None

    # Replace the original Reward column with the updated one
    df['Original_Reward'] = df['Reward']
    df['Reward'] = df['Updated_Reward']
    df = df.drop(columns=['Updated_Reward'])

    # Save the updated CSV file
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to: {output_csv}")

    # Print statistics
    valid_rewards = df['Reward'].dropna()
    num_updated = len(valid_rewards)

    print(f"Rewards updated: {num_updated}/{total_rows} ({num_updated / total_rows * 100:.2f}%)")

    if len(valid_rewards) > 0:
        print(f"Min reward: {valid_rewards.min()}")
        print(f"Max reward: {valid_rewards.max()}")
        print(f"Average reward: {valid_rewards.mean()}")

        # Count positive rewards
        positive_rewards = (valid_rewards > 0).sum()
        print(f"Positive rewards: {positive_rewards}/{num_updated} ({positive_rewards / num_updated * 100:.2f}%)")


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    try:
        # Load configurations
        norm_specs, ideal_specs = load_configs(args.config_path)

        # Import reward function
        cal_reward_func = import_reward_function(args.reward_func)

        # Update rewards
        update_rewards(args.input, args.output, cal_reward_func, norm_specs, ideal_specs, args.debug)

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()