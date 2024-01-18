import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env

num_episode = 10

ray.init()

def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs', sim_output=False, init_method='file')


register_env("AnalogDesignEnv_v0", env_creator)

ppo_config = {
    PPOConfig()
    .environment(env="AnalogDesignEnv_v0", clip_actions=True)
    .multi_agent(
            policies={"policy_1", "policy_2", "policy_3", "policy_4"},
            policy_mapping_fn=(lambda aid, episode, worker, **kw: f"policy_{aid[-1]}"),
            policies_to_train=["policy_1", "policy_2", "policy_3", "policy_4"],
        )
}

ppo = PPO(config = ppo_config)

checkpoint_path = ("/home/wuhan/ray_results/AnalogDesignEnv_v0/"
                   "PPO/PPO_AnalogDesignEnv_v0_5b5b4_00000_0_2024-01-16_21-15-45/checkpoint_000009/")
ppo.load_checkpoint(checkpoint_path)

env = RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs', sim_output=False, init_method='file')

for _ in range(num_episode):
    obs = env.reset()
    terminated = {"__all__": False}
    while not terminated["__all__"]:
        action = {agent_id: ppo.compute_single_action(observation) for agent_id, observation in obs.items()}
        obs, rew, terminateds, truncated, info = env.step(action)
        terminated = terminateds["__all__"]

ray.shutdown()
