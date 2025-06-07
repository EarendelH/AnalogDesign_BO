from ray.rllib.utils import check_env
from rllib_env_continous_psfascii import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv({'generalize': True,
                                    'max_step': 1,
                                    'netlist_folder_name': 'netlist_template_LFM',
                                    'specs_folder_name': 'sampled_specs_LFM',
                                    'config_folder_name': 'config_LFM_AutoGroup_14',
                                    'run_folder_name': 'run_LFM_Block',
                                    'sim_output': False,
                                    'init_method': 'random',
                                    'corner_sim': False,
                                    'dc_check': False,
                                    'region_extract': False,
                                    'dynamic_queue': False,
                                    'log_level': 'DEBUG',
                                    'reward_func': 'cal_reward_LFM',
                                    'continue_steps_enable': False
                                })

    check_env(env)
    print("Test passed")