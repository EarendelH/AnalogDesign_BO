from ray.rllib.algorithms.ppo import PPOConfig
import os
import ray
from rllib_env import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env

checkpoint_path = ("/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MA/AnalogDesignAuto_MultiAgent/custom_env/checkpoint_demo/checkpoint_000009")

ray.init()

def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs_validate', sim_output=False, init_method='file')

def policy_mapping_fn(agent_id):
    return f"policy_{agent_id[-1]}"

register_env("AnalogDesignEnv_v0", env_creator)

ppo_config = (
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

ppo = ppo_config.build()
ppo.restore(checkpoint_path)

policies = ['policy_1', 'policy_2', 'policy_3', 'policy_4']

for policy_name in policies:
    policy = ppo.get_policy(policy_name)
    export_path = f"/home/wuhan/Downloads/{policy_name}.onnx"
    policy.export_model(export_path, onnx=False)
    print(f"Model for {policy_name} exported to {export_path}")

ray.shutdown()
