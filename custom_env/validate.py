import concurrent.futures
import logging
import ray
import os
import random

from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env


def env_creator(env_config, validate_file_path, config_folder):
    return RllibAnalogDesignAutoEnv(generalize=False, specs_folder_name=validate_file_path,
                                    config_folder_name=config_folder, run_folder_name='run_test_validate',
                                    sim_output=False, init_method='file', action_mask=True, dc_check=True)


# def run_evaluation(checkpoint_file_path, validate_file_path, num_episodes=10):
#
#     register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config, validate_file_path))
#     env = env_creator({}, validate_file_path)
#
#     agent = Algorithm.from_checkpoint(checkpoint_file_path)
#
#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         done = {"__all__": False}
#         step = 0
#         count_terminated = 0
#         count_finished = 0
#         while not done["__all__"]:
#             action_dict = {}
#             for agent_id, agent_obs in obs.items():
#                 policy_id = f"policy_{agent_id[-1]}"
#                 action = agent.compute_single_action(agent_obs, policy_id=policy_id)
#                 action_dict[agent_id] = action
#             obs, rew, done, _, info = env.step(action_dict)
#             step += 1
#             if step > 1024:
#                 count_terminated += 1
#                 print(f"Episode {episode} terminated due to exceeding the maximum number of steps")
#                 break
#             if done["__all__"]:
#                 count_finished += 1
#                 print(f"Episode {episode} finished after {step} steps")
#
#         print(f"Total episodes: {count_finished + count_terminated}, Finished: {count_finished}, "
#               f"Terminated: {count_terminated}")
#
#     ray.shutdown()


def run_evaluation_single(checkpoint_file_path, validate_file_path, config_folder_name):
    register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config, validate_file_path,
                                                                      config_folder_name))
    env = env_creator({}, validate_file_path, config_folder_name)

    agent = Algorithm.from_checkpoint(checkpoint_file_path)

    obs, _ = env.reset()
    terminated_done = {"__all__": False}
    step = 0

    done_flag = None

    while not terminated_done["__all__"]:
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            policy_id = f"policy_{agent_id[-1]}"
            action = agent.compute_single_action(agent_obs, policy_id=policy_id)
            action_dict[agent_id] = action
        obs, rew, done, _, info = env.step(action_dict)
        step += 1
        if step >= 1024:
            done_flag = False
            break
        if terminated_done["__all__"]:
            done_flag = True

    return step, done_flag


def evaluate_specs(specs_file):
    validate_file_path = os.path.join(validate_file_folder, specs_file)
    step_count, success_flag = run_evaluation_single(checkpoint_path, validate_file_path, config_folder_name)
    print(f"Validation episode with file {specs_file} finished after {step_count} steps with success "
          f"flag {success_flag}")
    if success_flag:
        return 1
    else:
        return 0


if __name__ == "__main__":

    cpu_count = os.cpu_count()

    print("Init Validation")
    checkpoint_path = input("Enter the checkpoint path: ").strip()
    validate_file_folder = input("Enter the validation folder path: ").strip()
    config_folder_name = input("Enter the config folder name: ").strip()
    num_episode = input("Enter the number of episodes: ").strip()
    num_episode = int(num_episode)
    num_threads = input(f"Enter the number of threads, available CPU {cpu_count}: ").strip()
    num_threads = int(num_threads)

    ray.init(logging_level=logging.WARNING)

    episode_success_count = 0
    episode_fail_count = 0

    # Random select num_episode files from validate_file_folder and run run_evaluation_single

    file_list = os.listdir(validate_file_folder)
    selected_files = random.sample(file_list, num_episode)

    # for each file
    # Print the result for each file
    for file in selected_files:
        validate_file_path = os.path.join(validate_file_folder, file)
        step_count, success_flag = run_evaluation_single(checkpoint_path, validate_file_path, config_folder_name)
        print(f"Validation episode with file {file} finished after {step_count} steps with success flag {success_flag}")
        if success_flag:
            episode_success_count += 1
        else:
            episode_fail_count += 1
        print(f"Total episodes: {episode_success_count + episode_fail_count}, Finished: {episode_success_count}, "
              f"Terminated: {episode_fail_count}")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     results = executor.map(evaluate_specs, selected_files)
    #     for result in results:
    #         if result:
    #             episode_success_count += 1
    #         else:
    #             episode_fail_count += 1
    #         print(f"Total episodes: {episode_success_count + episode_fail_count}, Finished: {episode_success_count}, "
    #               f"Terminated: {episode_fail_count}")

    ray.shutdown()
