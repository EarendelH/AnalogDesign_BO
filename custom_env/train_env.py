import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCNet
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from rllib_env import RllibAnalogDesignAutoEnv


def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs', sim_output=False)


register_env("AnalogDesignEnv_v0", env_creator)

if __name__ == "__main__":
    env_name = "AnalogDesignEnv_v0"
    context = ray.init()
    print(context.dashboard_url)

    config = (
        PPOConfig()
        .environment(env="AnalogDesignEnv_v0", clip_actions=True)
        .rollouts(num_rollout_workers=36)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [256, 256, 256, 256, 256],
            }
        )
        .debugging(log_level="DEBUG")
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .multi_agent(
            policies={"policy_1", "policy_2", "policy_3", "policy_4"},
            policy_mapping_fn=(lambda aid, episode, worker, **kw: f"policy_{aid[-1]}"),
            policies_to_train=["policy_1", "policy_2", "policy_3", "policy_4"],
        )
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"training_iteration": 100},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="/home/wuhan/ray_results/" + env_name,
        config=config.to_dict(),
    )
