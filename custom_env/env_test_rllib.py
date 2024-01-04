from ray.rllib.utils import check_env
from rllib_env_warpper import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs')
    check_env(env)
    print("Test passed")