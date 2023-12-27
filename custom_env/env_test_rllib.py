from ray.rllib.utils import check_env
from rllib_env_warpper import RllibAnalogDesignAutoEnv

if __name__ == "__main__":
    env = RllibAnalogDesignAutoEnv(generalize=False, path='sampled_specs/1.yaml')
    check_env(env)
    print("Test passed")