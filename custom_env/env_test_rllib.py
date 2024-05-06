from ray.rllib.utils import check_env
from rllib_env import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Mohamed',
                                   netlist_folder_name='netlist_template_Mohamed', config_folder_name='config_Mohamed',
                                   run_folder_name='run_test', sim_output=False, init_method='file',
                                   dc_check=True)

    check_env(env)
    print("Test passed")
