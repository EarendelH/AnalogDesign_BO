import logging
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env_v3 import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def env_creator(env_config, validate_path):
    return RllibAnalogDesignAutoEnv(generalize=True, path=validate_path, sim_output=False, init_method='random',
                                    action_mask=True, init_dc_check=True)


def evaluate_episode(checkpoint_file_path, validate_path, episode_num):
    env = env_creator({}, validate_path)
    agent = Algorithm.from_checkpoint(checkpoint_file_path)
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
        if step > 1000:
            return {'episode': episode_num, 'steps': step, 'passed': False}
    return {'episode': episode_num, 'steps': step, 'passed': True}


def run_evaluation_parallel(checkpoint_file_path, validate_path, num_episodes=10, util_cpu=4):
    ray.init(logging_level=logging.WARNING)
    register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config, validate_path))

    passed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=util_cpu) as executor:
        futures = [executor.submit(evaluate_episode, checkpoint_file_path, validate_path, episode) for episode
                   in range(num_episodes)]
        for future in as_completed(futures):
            result = future.result()
            if result['passed']:
                passed_count += 1
            else:
                failed_count += 1
            print(f"Total Completed: {passed_count + failed_count}, Passed: {passed_count}, Failed: {failed_count}")

    ray.shutdown()


def get_user_thread_count():
    max_workers = os.cpu_count()
    print(f"Available threads: {max_workers}")
    cpu_num = input("Enter the number of threads to use: ").strip()
    return min(max_workers, int(cpu_num))


if __name__ == "__main__":
    print("Init Validation")
    checkpoint_path = input("Enter the checkpoint path: ").strip()
    path = input("Enter the folder path for validation: ").strip()
    num_episode = input("Enter the number of episodes: ").strip()
    num_episode = int(num_episode)
    num_workers = get_user_thread_count()
    run_evaluation_parallel(checkpoint_path, path, num_episode, num_workers)
