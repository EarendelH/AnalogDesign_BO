import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCNet
from ray.rllib.models import ModelCatalog
from torch import nn
from supersuit.multiagent_wrappers import pad_action_space_v0

from rllib_env_warpper import RllibAnalogDesignAutoEnv


class CustomFCNet(FCNet):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

    def forwad(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"])
        return model_out, state


def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs')


register_env("AnalogDesignEnv_v0", env_creator)

if __name__ == "__main__":
    ray.init()

    env_name = "AnalogDesignEnv_v0"

    ModelCatalog.register_custom_model("CustomFCNet", CustomFCNet)

    config = (
        PPOConfig()
        .environment(env="AnalogDesignEnv_v0", clip_actions=True)
        .rollouts(num_rollout_workers=2)
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
        )
        .debugging(log_level="ERROR")
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results" + env_name,
        config=config.to_dict(),
    )
