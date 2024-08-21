from ray.rllib.utils import check_env
from rllib_env_continous import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Cai',
                                   netlist_folder_name='netlist_template_Cai_YeLab',
                                   config_folder_name='config_Cai_Multi',
                                   run_folder_name='run_test', sim_output=False, init_method='random',
                                   dc_check=False, region_extract=True, dynamic_queue=False, log_level='DEBUG',
                                   max_step=10)

    check_env(env)
    print("Test passed")
