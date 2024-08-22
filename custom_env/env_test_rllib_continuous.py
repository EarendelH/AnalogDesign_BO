from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv({'generalize': True,
                                    'max_step': 10,
                                    'netlist_folder_name': 'sampled_specs_Buck',
                                    'specs_folder_name': 'sampled_specs_Buck',
                                    'config_folder_name': 'config_Buck',
                                    'run_folder_name': 'run_test',
                                    'sim_output': False,
                                    'init_method': 'random',
                                    'dc_check': False,
                                    'region_extract': False,
                                    'dynamic_queue': False,
                                    'log_level': 'INFO'
                                })

    check_env(env)
    print("Test passed")