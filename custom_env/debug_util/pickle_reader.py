import pickle
import json
import numpy as np

pickle_file = "../checkpoint_demo/checkpoint_000009/policies/policy_1/policy_state.pkl"

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

def convert_ndarray(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_ndarray(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray(v) for v in data]
    else:
        return data

converted_data = convert_ndarray(data)

with open('policy_state.json', 'w') as f:
    json.dump(data, f, indent=4)