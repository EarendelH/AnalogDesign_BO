import subprocess
import sys
import torch
import os

def set_max_process_limit():
    try:
        shell = subprocess.check_output('echo $0', shell=True).decode().strip()
        print(f"SHELL: {shell}")
        if 'bash' or '-bash' in shell:
            print("Setting max process limit to 40960")
            subprocess.call('ulimit -u 40960', shell=True)
        elif 'tcsh' or '-tcsh' in shell:
            print("Setting max process limit to 40960")
            subprocess.call('limit maxproc 40960', shell=True)
        else:
            print("Unknown shell. Not setting max process limit. Continuing? (y/n)")
            choice = input().strip().lower()
            if choice != 'n':
                print("Exiting")
                sys.exit(1)
    except subprocess.SubprocessError as e:
        print(f"Error setting max process limit: {e} and exiting")
        sys.exit(1)

set_max_process_limit()

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rllib_env import RllibAnalogDesignAutoEnv

num_cpu = int(os.cpu_count() * 0.8)

def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs', sim_output=False, init_method='file')


register_env("AnalogDesignEnv_v0", env_creator)

if __name__ == "__main__":
    env_name = "AnalogDesignEnv_v0"
    ray.init()

    # Restore or Initialize train
    restore_checkpoint = input("Restore from checkpoint? (y/n): ").strip().lower()
    checkpoint_path = None
    if restore_checkpoint == "y":
        checkpoint_path = input("Checkpoint path: ").strip()
        assert os.path.exists(checkpoint_path), "Checkpoint path does not exist"
        print(f"Restoring from checkpoint: {checkpoint_path}")

    # Typing Train Iterations
    train_iterations = input("Train iterations(Default 200): ").strip()
    if not train_iterations:
        train_iterations = 200
    else:
        try:
            train_iterations = int(train_iterations)
        except ValueError:
            print("Invalid input. Applying default value 200")
            train_iterations = 200

    config = (
        PPOConfig()
        .environment(env="AnalogDesignEnv_v0", clip_actions=True)
        .rollouts(num_rollout_workers=num_cpu)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [256, 256, 256, 256, 256],
            }
        )
        .debugging(log_level="DEBUG")
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
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
        restore=checkpoint_path if restore_checkpoint == "y" else None,
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir=f"{user_home_dir}/ray_results/{env_name}",
        config=config.to_dict(),
    )