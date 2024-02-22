import logging
import os

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env_v3 import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env
from ray.util.multiprocessing import Pool


def env_creator(env_config, validate_specs_path):
    return RllibAnalogDesignAutoEnv(generalize=True, path=validate_specs_path, sim_output=False, init_method='random',
                                    action_mask=True, init_dc_check=True)


def evaluate_episode(checkpoint_file_path, path, episode_num):
    register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config, path))
    env = env_creator({}, path)
    agent = Algorithm.from_checkpoint(checkpoint_file_path)
    obs, _ = env.reset()
    done = {"__all__": False}
    step = 0
    count = 0
    count_finished = 0
    count_terminated = 0
    while not done["__all__"]:
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            policy_id = f"policy_{agent_id[-1]}"
            action = agent.compute_single_action(agent_obs, policy_id=policy_id)
            action_dict[agent_id] = action
        obs, rew, done, _, info = env.step(action_dict)
        step += 1
        if done["__all__"]:
            count += 1
            count_finished += 1
            print(f"Total valid episodes: {count_finished}, finished episodes: {count_finished} "
                  f"and terminated episodes: {count_terminated}")
        if step > 1000:
            print(f"Episode {episode_num} terminated due to exceeding max steps")
            count += 1
            count_terminated += 1
            print(f"Total valid episodes: {count_finished}, finished episodes: {count_finished} "
                  f"and terminated episodes: {count_terminated}")
            break
    return f"Episode {episode_num} finished after {step} steps"


def run_evaluation_parallel(checkpoint_file_path, validate_specs_path, num_episodes, used_cpu):
    ray.init(logging_level=logging.WARNING)
    # Using a process pool to run evaluations in parallel
    with Pool(processes=used_cpu) as pool:
        results = pool.starmap(evaluate_episode, [(checkpoint_file_path, validate_specs_path, episode) for episode in
                                                  range(num_episodes)])
        for result in results:
            print(result)
    ray.shutdown()


if __name__ == "__main__":
    print("Init Validation")
    checkpoint_path = input("Enter the checkpoint path: ").strip()
    path = input("Enter the folder path for validation: ").strip()
    num_episode = input("Enter the number of episodes: ").strip()
    num_episode = int(num_episode)
    num_cpu_available = os.cpu_count()
    num_cpu = input(f"Enter the number of CPUs to use (Available num is {num_cpu_available}): ").strip()
    num_cpu = int(num_cpu)
    run_evaluation_parallel(checkpoint_path, path, num_episode, num_cpu)
