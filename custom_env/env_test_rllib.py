from ray.rllib.utils import check_env
from rllib_env import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, specs_folder_name='sampled_specs_v3',
                                   config_folder_name='config', run_dir_name='run_test', sim_output=False,
                                   init_method='half', action_mask=True, dc_check=True)

    check_env(env)
    print("Test passed")