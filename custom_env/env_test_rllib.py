from ray.rllib.utils import check_env
from rllib_env import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_ideal_source',
                                   config_folder_name='config_ideal_source', run_folder_name='run_test', sim_output=False,
                                   init_method='file', action_mask=True, dc_check=True)

    check_env(env)
    print("Test passed")
