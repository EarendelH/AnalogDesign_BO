from AnalogDesignAutoEnv import AnalogDesignEnv
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = AnalogDesignEnv(generalize=False, path='sampled_specs/1.yaml')
    parallel_api_test(env, num_cycles=1000)