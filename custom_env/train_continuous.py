import subprocess
import sys
import os
import argparse
import yaml
import torch
import logging

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from rllib_env_continous import RllibAnalogDesignAutoEnv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            "dc_check": get_user_input("Enable step DC check (True/False)", "True"),
            "dynamic_queue": get_user_input("Enable dynamic queue (True/False)", "True"),
            "restore_checkpoint": get_user_input("Restore from checkpoint? (True/False)", "False"),
            "checkpoint_path": None,  # To be conditionally updated
            "train_iterations": get_user_input("Train iterations(Default: 200)", "200"),
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
            settings["generalize"] = settings["generalize"] == "True"
            settings["sim_output"] = settings["sim_output"] == "True"
            settings["dc_check"] = settings["dc_check"] == "True"
            settings["dynamic_queue"] = settings["dynamic_queue"] == "True"
            settings["restore_checkpoint"] = settings["restore_checkpoint"] == "True"

        env_settings = {
            "generalize": settings["generalize"],
            "max_step": settings["max_step"],
            "netlist_folder_name": settings["netlist_folder_name"],
            "specs_folder_name": settings["specs_folder_name"],
            "config_folder_name": settings["config_folder_name"],
            "run_folder_name": settings["run_folder_name"],
            "sim_output": settings["sim_output"],
            "init_method": settings["init_method"],
            "dc_check": settings["dc_check"],
            "dynamic_queue": settings["dynamic_queue"],
        }

        # Environment initialization
        def env_creator(_):
            return RllibAnalogDesignAutoEnv(**env_settings)

        register_env("AnalogDesignEnv_v0", env_creator)

        env_name = "AnalogDesignEnv_v0"
        ray.init()

        # Restore or Initialize train
        restore_checkpoint = settings["restore_checkpoint"]
        if restore_checkpoint:
            checkpoint_path = settings["checkpoint_path"]
            assert os.path.exists(checkpoint_path), "Checkpoint path does not exist"
            logging.info(f"Restoring from checkpoint: {checkpoint_path}")

            # Use Algorithm.from_checkpoint() to restore the algorithm
            restored_algo = Algorithm.from_checkpoint(
                checkpoint=checkpoint_path
            )

            # Debug: Print information about the restored algorithm
            logging.info("Checkpoint restored successfully")
            logging.info(f"Restored algorithm type: {type(restored_algo).__name__}")

            # Get policy information
            if hasattr(restored_algo, 'workers') and restored_algo.workers:
                local_worker = restored_algo.workers.local_worker()
                if local_worker:
                    policies = local_worker.policy_map
                    logging.info(f"Restored policies: {list(policies.keys())}")
                else:
                    logging.warning("Local worker not available")
            else:
                logging.warning("Workers not available in restored algorithm")

            # Get the restored configuration
            logging.info("Configuration restored from checkpoint")
            restored_algo.train()
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


if __name__ == "__main__":
    main()
