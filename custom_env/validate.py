import ray
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env


def validate_checkpoint(checkpoint_path, num_episode):
    def env_creator(env_config):
        return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs', sim_output=False)

    env = register_env("AnalogDesignEnv_v0", env_creator)

    ray.init()
    episode_reward = 0

    algo = Algorithm.from_checkpoint(checkpoint_path)

    for i in range(num_episode):
        obs, _ = env.reset()
        action = algo.compute_actions(observations=obs)
        obs, reward, terminated, _, info = env.step(action)
        episode_reward += reward
        done = terminated
        print(f"Episode {i} reward: {episode_reward}")
        if done:
            print("Episode done!")

    ray.shutdown()

# Test Code
check_point = ("/home/wuhan/ray_results/AnalogDesignEnv_v0/PPO/"
               "PPO_AnalogDesignEnv_v0_6806e_00000_0_2024-01-12_13-06-55/checkpoint_000122/")
validate_checkpoint(check_point, 10)
