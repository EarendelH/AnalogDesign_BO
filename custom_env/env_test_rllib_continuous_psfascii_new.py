from ray.rllib.utils.pre_checks.env import check_multiagent_environments
from rllib_env_continous_psfascii import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv({'generalize': True,
                                    'max_step': 1,
                                    'netlist_folder_name': 'netlist_template_AXS',
                                    'specs_folder_name': 'sampled_specs_AXS',
                                    'config_folder_name': 'config_AXS_Simple',
                                    'run_folder_name': 'run_test',
                                    'sim_output': False,
                                    'init_method': 'file',
                                    'corner_sim': False,
                                    'dc_check': False,
                                    'region_extract': False,
                                    'dynamic_queue': False,
                                    'log_level': 'DEBUG',
                                    'reward_func': 'cal_reward_AXS',
                                    'continue_steps_enable': False
                                })

    check_multiagent_environments(env)
    print("Test passed")