from ray.rllib.utils import check_env
from rllib_env import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_Jianping_N65',
                                   netlist_folder_name='netlist_template_Jianping_N65_bk',
                                   config_folder_name='config_Jianping_N65', run_folder_name='run_test',
                                   sim_output=False, init_method='half', dc_check=True)

    check_env(env)
    print("Test passed")
