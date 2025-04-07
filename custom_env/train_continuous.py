
import sys
import os
import argparse
import yaml
import torch
import logging
import datetime
import time
import string
import random

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID
from typing import Dict
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR

from rllib_env_continous import RllibAnalogDesignAutoEnv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def generated_restore_id(length=6):
    characters = string.digits + string.ascii_letters
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string


def print_progress_table(id, result, iteration, total_time):
    # Define the columns and their formats
    columns = [
        ("ID", id, 6),
        ("Algorithm", "PPO_restored", 21),
        ("Iter", iteration, 4),
        ("Total Time", format_time(total_time), 10),
        ("Timesteps", result['timesteps_total'], 9),
        ("Reward Mean", result['episode_reward_mean'], 11, ".4f"),
        ("Reward Max", result['episode_reward_max'], 10, ".4f"),
        ("Reward Min", result['episode_reward_min'], 10, ".4f"),
        ("Ep Len Mean", result['episode_len_mean'], 11, ".4f")
    ]

    # Calculate the total width of the table
    total_width = sum(col[2] for col in columns) + len(columns) * 3 - 1

    # Create the header row
    header = "│ " + " │ ".join(f"{col[0]:<{col[2]}}" for col in columns) + " │"

    # Create the data row
    data = "│ " + " │ ".join(
        f"{col[1]:{col[2]}.{col[3]}}" if len(col) > 3 else f"{col[1]:<{col[2]}}"
        for col in columns
    ) + " │"

    # Print the table
    print("┌" + "─" * total_width + "┐")
    print(header)
    print("├" + "─" * total_width + "┤")
    print(data)
    print("└" + "─" * total_width + "┘")


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
            "cpu_usage": get_user_input(f"Enter use CPU num, total available CPU is {cpu_count}", "10"),
            "gpu_usage": get_user_input(f"Enter use GPU num, total available CPU is {gpu_count}", "0"),
            "algorithm": get_user_input("Algorithm (PPO/APPO/IMPALA)", "PPO"),
            "generalize": get_user_input("Enable generalization (True/False)", "True"),
            "num_agents": get_user_input("Number of agents(Default: 4)", "4"),
            "max_step": get_user_input("Max step(Default: 100)", "100"),
            "netlist_folder_name": get_user_input("Name of netlist folder", "netlist_template"),
            "specs_folder_name": get_user_input("Name of specs folder", "sampled_specs"),
            "config_folder_name": get_user_input("Name of config folder", "config"),
            "run_folder_name": get_user_input("Name of run folder", "run_test"),
            "sim_output": get_user_input("Enable simulation output (True/False)", "False"),
            "init_method": get_user_input("Initialization method (file/half/random/mixed)", "file"),
            "corner_sim": get_user_input("Enable corner simulation (True/False)", "False"),
            "dc_check": get_user_input("Enable step DC check (True/False)", "True"),
            "region_extract": get_user_input("Enable region extraction (True/False)", "True"),
            "dynamic_queue": get_user_input("Enable dynamic queue (True/False)", "True"),
            "log_level": get_user_input("Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", "INFO"),
            "reward_func": get_user_input("Reward function", "cal_reward_general"),
            "restore_checkpoint": get_user_input("Restore from checkpoint? (True/False)", "False"),
            "checkpoint_path": None,  # To be conditionally updated
            "train_iterations": get_user_input("Train iterations(Default: 200)", "200"),
            "continue_steps_enable": get_user_input("Enable continue steps (True/False)", "False"),
        }

        if settings["restore_checkpoint"] == "True":
            settings["checkpoint_path"] = get_user_input("Checkpoint path", "")
        confirm_flag = confirm_settings(settings)

    print(f"Configuration settings: {settings}")

    num_cpu = int(settings["cpu_usage"])
    num_gpu = int(settings["gpu_usage"])

    settings["train_iterations"] = int(settings["train_iterations"])

    if confirm_flag:
        if args.config_mode == 'interactive':
            # Convert string boolean values to Python boolean values
            for key in ["generalize", "sim_output", "corner_sim", "dc_check", "region_extract", "dynamic_queue",
                        "restore_checkpoint", "continue_steps_enable"]:
                settings[key] = settings[key].lower() == "true"

        # env_settings = {
        #     "generalize": settings["generalize"],
        #     "max_step": settings["max_step"],
        #     "netlist_folder_name": settings["netlist_folder_name"],
        #     "specs_folder_name": settings["specs_folder_name"],
        #     "config_folder_name": settings["config_folder_name"],
        #     "run_folder_name": settings["run_folder_name"],
        #     "sim_output": settings["sim_output"],
        #     "init_method": settings["init_method"],
        #     "dc_check": settings["dc_check"],
        #     "region_extract": settings["region_extract"],
        #     "dynamic_queue": settings["dynamic_queue"],
        # }

        # Environment initialization
        def env_creator(_):
            return RllibAnalogDesignAutoEnv({
                "generalize": settings["generalize"],
                "max_step": int(settings["max_step"]),
                "netlist_folder_name": settings["netlist_folder_name"],
                "specs_folder_name": settings["specs_folder_name"],
                "config_folder_name": settings["config_folder_name"],
                "run_folder_name": settings["run_folder_name"],
                "sim_output": settings["sim_output"],
                "init_method": settings["init_method"],
                "corner_sim": settings["corner_sim"],
                "dc_check": settings["dc_check"],
                "region_extract": settings["region_extract"],
                "dynamic_queue": settings["dynamic_queue"],
                "log_level": settings["log_level"],
                "reward_func": settings["reward_func"],
                "continue_steps_enable": settings["continue_steps_enable"]
            })

        register_env("AnalogDesignEnv_v0", env_creator)

        env_name = "AnalogDesignEnv_v0"
        ray.init()

        # Restore or Initialize train
        restore_checkpoint = settings["restore_checkpoint"]
        if settings["restore_checkpoint"]:
            policies = {f"policy_{i + 1}" for i in range(int(settings["num_agents"]))}
            policies_to_train = list(policies)
            policy_mapping_fn = lambda aid, episode, worker, **kwargs: f"policy_{int(aid[-1])}"

            config = (
                PPOConfig()
                .environment(env="AnalogDesignEnv_v0", clip_actions=True)
                .rollouts(num_rollout_workers=int(settings["cpu_usage"]))
                .training(
                    # train_batch_size=512,
                    train_batch_size=128,
                    lr=2e-4,
                    gamma=0.96,
                    lambda_=0.95,
                    use_gae=True,
                    clip_param=0.3,
                    grad_clip=None,
                    entropy_coeff=0.01,
                    vf_loss_coeff=0.25,
                    # sgd_minibatch_size=64,
                    sgd_minibatch_size=32,
                    num_sgd_iter=24,
                    model={
                        "fcnet_hiddens": [256, 256, 256, 256, 256],
                    }
                )
                .debugging(log_level="DEBUG")
                .framework("torch")
                .resources(num_gpus=int(settings["gpu_usage"]))
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=policies_to_train,
                )
            )

            checkpoint_path = settings["checkpoint_path"]
            assert os.path.exists(checkpoint_path), "Checkpoint path does not exist"
            logging.info(f"Attempting to restore from checkpoint: {checkpoint_path}")

            try:

                restore_id = generated_restore_id()

                new_log_dir = os.path.join(DEFAULT_RESULTS_DIR, env_name, f"restore_{restore_id}")
                os.makedirs(new_log_dir, exist_ok=True)

                def new_logger_creator(config):
                    return UnifiedLogger(config, new_log_dir, loggers=None)

                algo = config.build(logger_creator=new_logger_creator)
                logging.info("New algorithm instance built from configuration")
                logging.info(f"Algorithm: {algo}")

                algo.restore(checkpoint_path)
                logging.info(f"Algorithm state restored from checkpoint: {checkpoint_path}")

                restore_checkpoint_dir = os.path.join(new_log_dir, "checkpoints")
                os.makedirs(restore_checkpoint_dir, exist_ok=True)
                logging.info(f"New checkpoints will be saved in: {restore_checkpoint_dir}")
                logging.info(f"New logs will be saved in: {new_log_dir}")

                start_time = time.time()
                for iteration in range(int(settings["train_iterations"])):
                    result = algo.train()
                    logging.info(f"Detail data for iteration {iteration}: {result}")
                    total_time = time.time() - start_time

                    # print_progress_table(restore_id, result, iteration, total_time)

                    if iteration % 10 == 0:
                        # Checkpoint folder name with iteration number under checkpoint_path
                        checkpoint_index = iteration // 10
                        checkpoint_folder_iter = os.path.join(restore_checkpoint_dir, f"checkpoint_{checkpoint_index}")
                        checkpoint_result = algo.save(checkpoint_folder_iter)
                        new_checkpoint_path = checkpoint_result.checkpoint.path
                        logging.info(f"New checkpoint saved at iteration {iteration}: {new_checkpoint_path}")

                checkpoint_folder_final = os.path.join(restore_checkpoint_dir, "final_checkpoint")
                final_checkpoint_result = algo.save(checkpoint_folder_final)
                final_checkpoint_path = final_checkpoint_result.checkpoint.path
                logging.info(f"Final checkpoint saved: {final_checkpoint_path}")

            except Exception as e:
                logging.error(f"Error during checkpoint restoration or training: {e}")
                logging.exception("Detailed traceback:")
                sys.exit(1)

        if not restore_checkpoint:

            policies = {f"policy_{i + 1}" for i in range(settings["num_agents"])}
            policies_to_train = list(policies)
            policy_mapping_fn = lambda aid, episode, worker, **kwargs: f"policy_{int(aid[-1])}"

            logging.info("Starting new training session without checkpoint")
            # If not restoring, use the original configuration
            if settings["algorithm"] == "PPO":
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
                        num_sgd_iter=24,
                        model={
                            "fcnet_hiddens": [256, 256, 256, 256, 256],
                        }
                    )
                    .debugging(log_level="DEBUG")
                    .framework("torch")
                    .resources(num_gpus=num_gpu)
                    .multi_agent(
                        policies=policies,
                        policy_mapping_fn=policy_mapping_fn,
                        policies_to_train=policies_to_train,
                    )
                )
            elif settings["algorithm"] == "APPO":
                config = (
                    APPOConfig()
                    .environment(env="AnalogDesignEnv_v0", clip_actions=True)
                    .rollouts(num_rollout_workers=num_cpu)
                    .training(
                        train_batch_size=512,
                        lr=2e-4,
                        gamma=0.96,
                        lambda_=0.95,
                        use_gae=True,
                        clip_param=0.3,
                        grad_clip=40,
                        entropy_coeff=0.01,
                        vf_loss_coeff=0.5,
                        vtrace=True,
                        use_kl_loss=False,
                        num_sgd_iter=1,
                        minibatch_buffer_size=1,
                        replay_proportion=0.2,
                        replay_buffer_num_slots=1000,
                        broadcast_interval=1,
                        model={
                            "fcnet_hiddens": [256, 256, 256, 256, 256],
                        }
                    )
                    .debugging(log_level="DEBUG")
                    .framework("torch")
                    .resources(num_gpus=num_gpu)
                    .multi_agent(
                        policies=policies,
                        policy_mapping_fn=policy_mapping_fn,
                        policies_to_train=policies_to_train,
                    )
                )
            elif settings["algorithm"] == "IMPALA":
                config = (
                    ImpalaConfig()
                    .environment(env="AnalogDesignEnv_v0", clip_actions=True)
                    .rollouts(num_rollout_workers=num_cpu)
                    .training(
                        train_batch_size=512,
                        minibatch_size="auto",
                        num_sgd_iter=1,
                        lr=2e-4,
                        gamma=0.96,
                        grad_clip=40.0,
                        vf_loss_coeff=0.5,
                        entropy_coeff=0.01,
                        vtrace=True,
                        learner_queue_size=3,
                        broadcast_interval=1,
                        model={
                            "fcnet_hiddens": [256, 256, 256, 256, 256],
                        },
                    )
                    .resources(num_gpus=num_gpu)
                    .debugging(log_level="DEBUG")
                    .framework("torch")
                    .multi_agent(
                        policies=policies,
                        policy_mapping_fn=policy_mapping_fn,
                        policies_to_train=policies_to_train,
                    )
                )
            elif settings["algorithm"] == "SAC":
                config = (
                    SACConfig()
                    .environment(env="AnalogDesignEnv_v0", clip_actions=False)
                    .rollouts(num_rollout_workers=num_cpu)
                    .training(
                        twin_q=True,
                        q_model_config={
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        },
                        policy_model_config={
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        },
                        tau=5e-3,
                        initial_alpha=1.0,
                        target_entropy="auto",
                        n_step=1,
                        train_batch_size=512,
                        num_steps_sampled_before_learning_starts=1024,
                        target_network_update_freq=0,
                        grad_clip=40,
                    )
                    .resources(num_gpus=num_gpu)
                    .debugging(log_level="DEBUG")
                    .framework("torch")
                    .multi_agent(
                        policies=policies,
                        policy_mapping_fn=policy_mapping_fn,
                        policies_to_train=policies_to_train,
                    )
                )
            else:
                raise ValueError(f"Unsupported algorithm: {settings['algorithm']}")

            # Typing Train Iterations
            train_iterations = settings["train_iterations"]

            logging.info("Starting training process...")

            user_home_dir = os.path.expanduser("~")

            # Run the training

            alg_name = settings["algorithm"]

            tune.run(
                alg_name,
                name=alg_name,
                stop={"training_iteration": train_iterations},
                checkpoint_freq=25,
                checkpoint_at_end=True,
                local_dir=f"{user_home_dir}/ray_results/{env_name}",
                config=config.to_dict() if isinstance(config, PPOConfig) else config,
            )
    else:
        print("Configuration not confirmed. Training aborted.")


def print_model_structure(model):
    logging.info("Model structure:")
    for name, param in model.named_parameters():
        logging.info(f"  {name}: {param.shape}")


if __name__ == "__main__":
    main()
