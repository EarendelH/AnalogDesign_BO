from ray.rllib.utils import check_env
from rllib_env_v3 import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs_v3', sim_output=False, init_method='random',
                                   action_mask=True, init_dc_check=True)

    check_env(env)
    print("Test passed")