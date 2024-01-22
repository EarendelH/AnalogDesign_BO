import pickle
import json

pickle_file = "../checkpoint_demo/checkpoint_000009/policies/policy_1/policy_state.pkl"

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

with open('policy_state.json', 'w') as f:
    json.dump(data, f, indent=4)