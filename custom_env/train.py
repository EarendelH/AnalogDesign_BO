import subprocess
import sys
import os
import argparse
import yaml
import torch

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rllib_env import RllibAnalogDesignAutoEnv


def set_max_process_limit():
    try:
        shell = subprocess.check_output('echo $0', shell=True).decode().strip()
        print(f"SHELL: {shell}")
        if 'bash' or '-bash' in shell:
            print("Setting max process limit to 409600")
            subprocess.call('ulimit -u 409600', shell=True)
        elif 'tcsh' or '-tcsh' in shell:
            print("Setting max process limit to 409600")
            subprocess.call('limit maxproc 409600', shell=True)
        else:
            print("Unknown shell. Not setting max process limit. Continuing? (y/n): ")
            choice = input().strip().lower()
            if choice != 'n':
                print("Exiting")
                sys.exit(1)
    except subprocess.SubprocessError as e:
        print(f"Error setting max process limit: {e} and exiting")
        sys.exit(1)


def get_user_input(prompt, default_value):
    """Collects user input or uses the default value if input is empty."""
    user_input = input(f"{prompt} (Default: {default_value}): ").strip()
    return user_input if user_input else default_value


def confirm_settings(settings):
    """Displays settings for user confirmation before proceeding with training."""
    print("\nHere are your configuration settings:")
    for k, v in settings.items():
        print(f"{k}: {v}")
    confirm = input("\nConfirm to start training? (y/n): ").lower().strip()
    return confirm == "y"


def main():
    """Main function to run the script."""
    settings = {}
    confirm_flag = None
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()

    parser = argparse.ArgumentParser(description="Train Analog Design AutoRL Environment")
    parser.add_argument('--config_mode', type=str, choices=['interactive', 'file'], default='interactive',
                        help="Mode for configuration settings")
    parser.add_argument('--config_file', type=str, help="Path to configuration file")
    args = parser.parse_args()

    print("Config Mode: ", args.config_mode)

    if args.config_mode == 'file':
        if args.config_file is None:
            print("Configuration file not provided. Exiting.")
            sys.exit(1)
        with open(args.config_file, 'r') as file:
            settings = yaml.safe_load(file)
        confirm_flag = True
    if args.config_mode == 'interactive':
        settings = {
            "max_process_limit": get_user_input("Set max process limit? (True/False)", "False"),
            "cpu_usage": get_user_input(f"Enter use CPU num, total available CPU is {cpu_count}"
                                                   , "10"),
            "gpu_usage": get_user_input(f"Enter use GPU num, total available CPU is {gpu_count}", "0"),
            "generalize": get_user_input("Enable generalization (True/False)", "True"),
            "specs_folder_name": get_user_input("Name of specs folder", "sampled_specs"),
            "config_folder_name": get_user_input("Name of config folder", "config"),
            "run_folder_name": get_user_input("Name of run folder", "run_test"),
            "sim_output": get_user_input("Enable simulation output (True/False)", "False"),
            "init_method": get_user_input("Initialization method (File/Half/Random)", "file"),
            "action_mask": get_user_input("Enable action mask (True/False)", "True"),
            "dc_check": get_user_input("Enable step DC check (True/False)", "True"),
            "restore_checkpoint": get_user_input("Restore from checkpoint? (True/False)", "False"),
            "checkpoint_path": None,  # To be conditionally updated
            "train_iterations": get_user_input("Train iterations(Default: 200)", "200"),
        }

        if settings["restore_checkpoint"]:
            settings["checkpoint_path"] = get_user_input("Checkpoint path", "")
        confirm_flag = confirm_settings(settings)

    print(f"Configuration settings: {settings}")

    if settings["max_process_limit"]:
        set_max_process_limit()

    num_cpu = int(settings["cpu_usage"])
    num_gpu = int(settings["gpu_usage"])

    settings["train_iterations"] = int(settings["train_iterations"])

    if confirm_flag:
        if args.config_mode == 'interactive':
            # Convert string boolean values to Python boolean values
            settings["max_process_limit"] = settings["max_process_limit"] == "True"
            settings["generalize"] = settings["generalize"] == "True"
            settings["sim_output"] = settings["sim_output"] == "True"
            settings["action_mask"] = settings["action_mask"] == "True"
            settings["dc_check"] = settings["dc_check"] == "True"
            settings["restore_checkpoint"] = settings["restore_checkpoint"] == "True"

        env_settings = {
            "generalize": settings["generalize"],
            "specs_folder_name": settings["specs_folder_name"],
            "config_folder_name": settings["config_folder_name"],
            "run_folder_name": settings["run_folder_name"],
            "sim_output": settings["sim_output"],
            "init_method": settings["init_method"],
            "action_mask": settings["action_mask"],
            "dc_check": settings["dc_check"],
        }

        # Environment initialization
        def env_creator(_):
            return RllibAnalogDesignAutoEnv(**env_settings)

        register_env("AnalogDesignEnv_v0", env_creator)

        # Configuration and launching of the training process would go here
        # Similar to the previously described code for setting up and running the training

        env_name = "AnalogDesignEnv_v0"
        ray.init()

        # Restore or Initialize train
        restore_checkpoint = settings["restore_checkpoint"]
        checkpoint_path = None
        if restore_checkpoint == "True":
            checkpoint_path = settings["checkpoint_path"]
            assert os.path.exists(checkpoint_path), "Checkpoint path does not exist"
            print(f"Restoring from checkpoint: {checkpoint_path}")

        # Typing Train Iterations
        train_iterations = settings["train_iterations"]

        print("Starting training process...")

        config = (
            PPOConfig()
            .environment(env="AnalogDesignEnv_v0", clip_actions=True)
            .rollouts(num_rollout_workers=num_cpu)
            .training(
                train_batch_size=512,
                lr=2e-4,
                gamma=0.96,
                lambda_=0.95,
                use_gae=True,
                clip_param=0.3,
                grad_clip=None,
                entropy_coeff=0.01,
                vf_loss_coeff=0.25,
                sgd_minibatch_size=64,
                num_sgd_iter=10,
                model={
                    "fcnet_hiddens": [256, 256, 256, 256, 256],
                }
            )
            .debugging(log_level="DEBUG")
            .framework("torch")
            .resources(num_gpus=num_gpu)
            .multi_agent(
                policies={"policy_1", "policy_2", "policy_3", "policy_4"},
                policy_mapping_fn=(lambda aid, episode, worker, **kw: f"policy_{aid[-1]}"),
                policies_to_train=["policy_1", "policy_2", "policy_3", "policy_4"],
            )
        )

        user_home_dir = os.path.expanduser("~")

        tune.run(
            "PPO",
            name="PPO",
            stop={"training_iteration": train_iterations},
            restore=checkpoint_path if restore_checkpoint == "True" else None,
            checkpoint_freq=25,
            checkpoint_at_end=True,
            local_dir=f"{user_home_dir}/ray_results/{env_name}",
            config=config.to_dict(),
        )

    else:
        print("Configuration not confirmed. Training aborted.")


if __name__ == "__main__":
    main()
