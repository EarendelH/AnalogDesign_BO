from ray.rllib.algorithms.ppo import PPOConfig
import os
import ray
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env

checkpoint_path = ("../checkpoint_demo/checkpoint_000199")

ray.init()

def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='../sampled_specs_validate', sim_output=False, init_method='file')

register_env("AnalogDesignEnv_v0", env_creator)

ppo_config = PPOConfig().environment("AnalogDesignEnv_v0")
ppo = ppo_config.build()
ppo.restore(checkpoint_path)

policies = ['policy_1', 'policy_2', 'policy_3', 'policy_4']

for policy_name in policies:
    policy = ppo.get_policy(policy_name)
    export_path = f"~/Downloads/{policy_name}.onnx"
    policy.export_model(export_path, onnx=False)
    print(f"Model for {policy_name} exported to {export_path}")

ray.shutdown()
