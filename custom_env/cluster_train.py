import subprocess
import os
import socket
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from rllib_env_v4 import RllibAnalogDesignAutoEnv

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
            print("Unknown shell. Not setting max process limit. Continuing? (y/n)")
            choice = input().strip().lower()
            if choice != 'n':
                print("Exiting")
                sys.exit(1)
    except subprocess.SubprocessError as e:
        print(f"Error setting max process limit: {e} and exiting")
        sys.exit(1)


def get_cpu_cores():
    return os.cpu_count()


def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs_v4', sim_output=False, init_method='file',
                                    action_mask=True, init_dc_check=False)


def start_training(num_cpu, checkpoint_path, train_iterations):
    print(f"Starting training with {num_cpu} CPU cores")
    register_env("AnalogDesignEnv_v0", env_creator)
    env_name = "AnalogDesignEnv_v0"
    ray.init(ignore_reinit_error=True, num_cpus=num_cpu)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
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
                "fcnet_hiddens": [512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
            }
        )
        .framework("torch")
    )

    user_home_dir = os.path.expanduser("~")
    tune.run(
        "PPO",
        name="PPO",
        stop={"training_iteration": train_iterations},
        restore=checkpoint_path if checkpoint_path else None,
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir=f"{user_home_dir}/ray_results/{env_name}",
        config=config.to_dict(),
    )


def start_ray_master(default_password='password'):
    print(f"Total available CPU cores: {get_cpu_cores()}")
    num_cpus = int(input("Enter the number of CPU cores to use: "))
    input_password = input(f"Please enter the redis password to continue [{default_password}]: ") or default_password
    ray.init(num_cpus=num_cpus, _node_ip_address=socket.gethostbyname(socket.gethostname()),
             _redis_password=input_password)
    print(f"Ray Master is running at {ray.get_webui_url()}")
    print("Waiting for slave nodes to join the cluster...")

    while True:
        nodes = ray.nodes()
        print(f"Current nodes in the cluster: {len(nodes)}")
        # Check if new slave nodes have joined, if so, print their information
        for node in nodes:
            if node['NodeManagerAddress'] != socket.gethostbyname(socket.gethostname()):
                print(f"New slave node joined: {node}")
        # Simple mechanism to break the loop, in real scenario, you might want to use a more robust condition
        if input("Press 'q' to quit waiting for new nodes: ").lower() == 'q':
            break
        # Start training process after all nodes have joined

        # Restore or Initialize train
        restore_checkpoint = input("Restore from checkpoint? (y/n): ").strip().lower()
        input_checkpoint_path = None
        if restore_checkpoint == "y":
            input_checkpoint_path = input("Checkpoint path: ").strip()
            assert os.path.exists(input_checkpoint_path), "Checkpoint path does not exist"
            print(f"Restoring from checkpoint: {input_checkpoint_path}")

        # Typing Train Iterations
        set_train_iterations = input("Train iterations(Default 200): ").strip()
        if not set_train_iterations:
            set_train_iterations = 200
        else:
            try:
                set_train_iterations = int(set_train_iterations)
            except ValueError:
                print("Invalid input. Applying default value 200")
                set_train_iterations = 200

    start_training(num_cpus, input_checkpoint_path, set_train_iterations)


def join_ray_cluster():
    master_ip = input("Please enter the master node IP address: ")
    redis_password = input("Please enter the redis password: ")
    print(f"Total available CPU cores: {get_cpu_cores()}")
    num_cpus = int(input("Enter the number of CPU cores to use: "))
    ray.init(address=f'ray://{master_ip}:6379', _redis_password=redis_password, num_cpus=num_cpus)
    print("Successfully joined the Ray cluster.")


def main():
    set_max_process_limit()
    role = input("Enter the role (master/slave): ").strip().lower()
    if role == 'master':
        start_ray_master()
    elif role == 'slave':
        join_ray_cluster()
    else:
        print("Invalid role. Exiting")
        sys.exit(1)


if __name__ == "__main__":
    main()
