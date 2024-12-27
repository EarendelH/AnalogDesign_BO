from ray.rllib.utils import check_env
from rllib_env_continous_psfascii import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv({'generalize': True,
                                    'max_step': 1,
                                    'netlist_folder_name': 'netlist_template_DRMOS',
                                    'specs_folder_name': 'sampled_specs_DRMOS',
                                    'config_folder_name': 'config_DRMOS',
                                    'run_folder_name': 'run_test',
                                    'sim_output': False,
                                    'init_method': 'file',
                                    'corner_sim': False,
                                    'dc_check': False,
                                    'region_extract': False,
                                    'dynamic_queue': False,
                                    'log_level': 'INFO',
                                    'reward_func': 'cal_reward_DRMOS',
                                    'continue_steps_enable': False
                                })

    check_env(env)
    print("Test passed")