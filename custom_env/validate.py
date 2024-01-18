import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env

num_episode = 10

ray.init()

def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs', sim_output=False, init_method='file')

def run_evaluation(checkpoint_path, num_episodes=10):
    ray.init()
    register_env("AnalogDesignEnv_v0", env_creator)
    env = env_creator({})

    config = PPOConfig().environment(env="AnalogDesignEnv_v0").to_dict()
    agent = Algorithm.from_checkpoint(checkpoint_path, config=config)

    for episode in range(num_episodes):
        obs = env.reset()
        done = {"__all__": False}
        while not done["__all__"]:
            action_dict = {}
            for agent_id, agent_obs in obs.items():
                policy_id = f"policy_{agent_id[-1]}"
                action = agent.compute_single_action(agent_obs, policy_id=policy_id)
                action_dict[agent_id] = action
            obs, rew, done, _, info = env.step(action_dict)

        print(f"Episode reward: {rew} at episode {episode}")

    ray.shutdown()

if __name__ == "__main__":
    checkpoint_path = "/home/wuhan/ray_results/AnalogDesignEnv_v0/PPO/PPO_AnalogDesignEnv_v0_5b5b4_00000_0_2024-01-16_21-15-45/checkpoint_000009/"
    num_episode = 10
    run_evaluation(checkpoint_path, num_episode)
