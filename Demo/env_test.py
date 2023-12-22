from custom_environment import CustomEnvironment
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = CustomEnvironment()
    try:
        parallel_api_test(env, num_cycles=5)
        print("Test passed")
    except Exception as e:
        print("Test failed")
        print(e)