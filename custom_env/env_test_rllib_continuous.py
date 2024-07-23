from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Lab_N65',
                                   netlist_folder_name='netlist_template_Lab_N65',
                                   config_folder_name='config_Lab_N65',
                                   run_folder_name='run_test', sim_output=False, init_method='half',
                                   dc_check=True, dynamic_queue=False, log_level='INFO')

    check_env(env)
    print("Test passed")
