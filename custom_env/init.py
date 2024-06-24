import ray
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_env_continous import RllibAnalogDesignAutoEnv
from ray.tune.registry import register_env


def step_inference(policies, env, max_step=20):

    observations, _ = env.reset()
    actions = None

    for step in range(max_step):
        actions = {}
        for agent_id, agent_obs in observations.items():
            policy_id = f"policy_{agent_id[-1]}"
            action = policies.compute_single_action(agent_obs, policy_id=policy_id)
            actions[agent_id] = action

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if any(reward > 0 for reward in rewards.values()):
            print(f"Generate init point successfully at step {step} with rewards {rewards}")
            break

        if terminated["__all__"] or truncated["__all__"]:
            print(f"Generate init point failed at step {step} due to termination or truncation with obs {observations}")
            break

    return actions


def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Jianping_DC',
                                    netlist_folder_name='netlist_template_Jianping_N65_bk_cap',
                                    config_folder_name='config_Jianping_N65_DC', run_folder_name='run_init',
                                    sim_output=False, init_method='random', dc_check=True, log_level='WARNING')


def main(checkpoint_path):
    ray.init()

    register_env("AnalogDesignEnv_v0", lambda env_config: env_creator(env_config))

    env = env_creator({})

    agents = Algorithm.from_checkpoint(checkpoint_path)

    valid_actions = step_inference(agents, env, max_step=20)

    print(f"Valid actions: {valid_actions}")

    env.close()


if __name__ == "__main__":
    checkpoint_path = input("Enter the checkpoint path: ").strip()
    main(checkpoint_path)
