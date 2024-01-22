from ray.rllib.algorithms.ppo import PPOConfig
import os
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env

checkpoint_path = ("../checkpoint_demo/checkpoint_000199")

def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs_validate', sim_output=False, init_method='file')

register_env("AnalogDesignEnv_v0", env_creator)

ppo_config = PPOConfig().environment("AnalogDesignEnv_v0")

ppo = ppo_config.build()

ppo_policy = ppo.get_policy()

export_path = "~/Downloads/ONNX"
ppo_policy.export_model(export_path, onnx=True)
