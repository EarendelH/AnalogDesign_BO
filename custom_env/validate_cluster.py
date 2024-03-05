import logging
import ray
import os

from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env


def env_creator(env_config, validate_dir_name, config_dir_name, run_folder_name):
    return RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name=validate_dir_name,
                                    config_folder_name=config_dir_name, run_folder_name=run_folder_name,
                                    sim_output=False, init_method='file', action_mask=True, dc_check=True)


def run_evaluation(checkpoint_file_path, validate_file_path, config_dir_name, run_folder_name, num_episodes):
    ray.init()

    register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config, validate_file_path,
                                                                      config_dir_name, run_folder_name))
    env = env_creator({}, validate_file_path, config_dir_name, run_folder_name)

    agent = Algorithm.from_checkpoint(checkpoint_file_path)

    count_terminated = 0
    count_finished = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        step = 0
        while not done["__all__"]:
            action_dict = {}
            for agent_id, agent_obs in obs.items():
                policy_id = f"policy_{agent_id[-1]}"
                action = agent.compute_single_action(agent_obs, policy_id=policy_id)
                action_dict[agent_id] = action
            obs, rew, done, _, info = env.step(action_dict)
            step += 1
            if step > 1024:
                count_terminated += 1
                print(f"Episode {episode} terminated due to exceeding the maximum number of steps")
                break
            if done["__all__"]:
                count_finished += 1
                print(f"Episode {episode} finished after {step} steps")

        print(f"Total episodes: {count_finished + count_terminated}, Finished: {count_finished}, "
              f"Terminated: {count_terminated}")

    ray.shutdown()


if __name__ == "__main__":
    print("Init Validation")
    # Get the script path
    checkpoint_path = "/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/checkpoint_demo/checkpoint_f8942_v4"
    validate_file_folder = "sampled_specs_validate_v4"
    config_folder_name = "config_v4"
    run_dir_name = "run_validate"
    num_episode = 100

    run_evaluation(checkpoint_path, validate_file_folder, config_folder_name, run_dir_name, num_episode)
