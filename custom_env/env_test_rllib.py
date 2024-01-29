from ray.rllib.utils import check_env
from rllib_env import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(action_mask=True, generalize=True, path='sampled_specs', sim_output=False,
                                   init_method='random')
    check_env(env)
    print("Test passed")