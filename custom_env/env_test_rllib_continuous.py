from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Lab_N65',
                                   netlist_folder_name='netlist_template_Lab_N65_Yelab',
                                   config_folder_name='config_Jianping_N65_C',
                                   run_folder_name='run_test', sim_output=False, init_method='half',
                                   dc_check=True, dynamic_queue=True, log_level='DEBUG')

    check_env(env)
    print("Test passed")
