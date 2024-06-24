import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
import numpy as np
from rllib_env_continous import RllibAnalogDesignAutoEnv

ray.init()


def load_policies(checkpoint_path, policy_ids):
    algo = Algorithm.from_checkpoint(checkpoint_path)
    return {pid: algo.get_policy(pid) for pid in policy_ids}


def step_inference(policies, env, max_step=20):

    observations, infos = env.reset()
    actions = None

    for step in range(max_step):
        actions = {}
        for agent_id, obs in observations.items():
            policy = policies[agent_id]
            action = policy.compute_single_action(obs)
            actions[agent_id] = action

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if any(reward > 0 for reward in rewards.values()):
            print(f"Generate init point successfully at step {step} with rewards {rewards}")
            break

        if terminated["__all__"] or truncated["__all__"]:
            print(f"Generate init point failed at step {step} due to termination or truncation with obs {observations}")
            break

    return actions


def main(checkpoint_path):

    checkpoint_ids = ['policy_1']

    policies = load_policies(checkpoint_path, checkpoint_ids)

    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Jianping_DC',
                                   netlist_folder_name='netlist_template_Jianping_N65_bk_cap',
                                   config_folder_name='config_Jianping_N65_DC',
                                   run_folder_name='run_init', sim_output=False, init_method='random',
                                   dc_check=True, log_level='DEBUG')

    valid_actions = step_inference(policies, env, max_step=20)

    print(f"Valid actions: {valid_actions}")

    env.close()


if __name__ == "__main__":
    checkpoint_path = input("Enter the checkpoint path: ").strip()
    main(checkpoint_path)
