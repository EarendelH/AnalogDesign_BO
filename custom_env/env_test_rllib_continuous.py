from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv({'generalize': True,
                                    'max_step': 1,
                                    'netlist_folder_name': 'netlist_template_AXS',
                                    'specs_folder_name': 'sampled_specs_AXS_no_Mismatch',
                                    'config_folder_name': 'config_AXS_no_Mismatch',
                                    'run_folder_name': 'run_test',
                                    'sim_output': False,
                                    'init_method': 'random',
                                    'corner_sim': False,
                                    'dc_check': False,
                                    'region_extract': False,
                                    'dynamic_queue': False,
                                    'log_level': 'DEBUG',
                                    'reward_func': 'cal_reward_Haoqiang',
                                    'continue_steps_enable': False
                                })

    check_env(env)
    print("Test passed")