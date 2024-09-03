from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv({'generalize': True,
                                    'max_step': 10,
                                    'netlist_folder_name': 'netlist_template_Haoqiang',
                                    'specs_folder_name': 'sampled_specs_Haoqiang',
                                    'config_folder_name': 'config_Haoqiang',
                                    'run_folder_name': 'run_test',
                                    'sim_output': False,
                                    'init_method': 'random',
                                    'dc_check': False,
                                    'region_extract': True,
                                    'dynamic_queue': False,
                                    'log_level': 'DEBUG',
                                    'reward_func': 'cal_reward_LDO'
                                })

    check_env(env)
    print("Test passed")