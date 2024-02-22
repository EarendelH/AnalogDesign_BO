import logging
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env_v3 import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env


def env_creator(env_config, validate_path):
    return RllibAnalogDesignAutoEnv(generalize=True, path=validate_path, sim_output=False, init_method='random',
                                    action_mask=True, init_dc_check=True)


def run_evaluation(checkpoint_file_path, path, num_episodes=10):
    ray.init(logging_level=logging.WARNING)

    register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config, path))
    env = env_creator({}, path)

    agent = Algorithm.from_checkpoint(checkpoint_file_path)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        step = 0
        count_terminated = 0
        count_finished = 0
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
    checkpoint_path = input("Enter the checkpoint path: ").strip()
    path = input("Enter the folder path for validation: ").strip()
    num_episode = input("Enter the number of episodes: ").strip()
    num_episode = int(num_episode)
    run_evaluation(checkpoint_path, path, num_episode)