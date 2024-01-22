import pickle
import json
import numpy as np

pickle_file = "../checkpoint_demo/checkpoint_000009/policies/policy_1/policy_state.pkl"

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

print(data)