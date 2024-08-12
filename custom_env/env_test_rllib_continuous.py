from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Cai',
                                   netlist_folder_name='netlist_template_Cai',
                                   config_folder_name='config_Cai_Single',
                                   run_folder_name='run_test', sim_output=False, init_method='file',
                                   dc_check=False, dynamic_queue=False, log_level='INFO', max_step=10)

    check_env(env)
    print("Test passed")
