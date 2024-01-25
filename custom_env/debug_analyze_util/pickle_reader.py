import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style

# Load data from the provided pickle file
pickle_file = "/Users/hanwu/Downloads/run_test/202401251122043275.pkl"
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Processing initial parameters from the data
init_param = data['initial_data']['init_param']
init_param = dict(init_param)
updated_params = [step['updated_param'] for step in data['steps_data']]
updated_params = [dict(params) for params in updated_params]

# Aggregating parameter names from initial and updated parameters
param_names = set(init_param.keys())
for params in updated_params:
    param_names.update(params.keys())

# Creating a dictionary to hold the history of each parameter's value
params_dict = {param: [init_param.get(param, None)] for param in param_names}
for i, params in enumerate(updated_params, start=1):
    for param in param_names:
        params_dict[param].append(params.get(param, params_dict[param][i-1]))

# Creating a DataFrame to display the parameter values at each step
df = pd.DataFrame(params_dict)
df.insert(0, 'Step', ['Initial'] + [f'Step {i}' for i in range(1, len(updated_params) + 1)])
df.set_index('Step', inplace=True)

# Printing the DataFrame
print(df.T)

# Function to flatten the simulation result structure
def flatten_result(sim_result):
    flat_result = {}
    for category, metrics in sim_result.items():
        for metric, value in metrics.items():
            flat_result[f"{metric}"] = value
    return flat_result

# Flattening the simulation results
initial_sim_result = flatten_result(data['initial_data']['sim_result'])
sim_results = [flatten_result(step['sim_result']) for step in data['steps_data']]

# Aggregating all metric names from the simulation results
all_metrics = set(initial_sim_result.keys())
for result in sim_results:
    all_metrics.update(result.keys())

# Creating a dictionary to hold the history of each metric's value
metrics_data = {metric: [initial_sim_result.get(metric, np.nan)] for metric in all_metrics}
for result in sim_results:
    for metric in all_metrics:
        metrics_data[metric].append(result.get(metric, np.nan))

# Extracting ideal specifications
ideal_specs = data['initial_data']['ideal_specs']
ideal_values = {key: value['value'] for key, value in ideal_specs.items()}

# Generating an index for each step, including the initial step
index = np.arange(len(sim_results) + 1)

# Setting up subplots for each metric to plot their values over the steps
n_metrics = len(metrics_data)
fig, axs = plt.subplots(1, n_metrics, figsize=(8 * n_metrics, 6), constrained_layout=True)

print(plt.style.available)

# Plotting the values of each metric and their ideal specifications if available
for i, (metric, values) in enumerate(metrics_data.items()):
    axs[i].plot(index, values, '-o', label=metric)
    if metric in ideal_values:
        ideal_value = ideal_values[metric]
        axs[i].axhline(y=ideal_value, color='red', linestyle='--', label='ideal specs')
        axs[i].text(0.5, ideal_value, 'ideal specs', color='red', va='bottom', ha='center')
    axs[i].set_title(f"{metric}, ideal value: {ideal_values.get(metric)}")
    axs[i].set_xlabel('Step Number')
    axs[i].set_ylabel(metric)
    axs[i].legend(loc='upper left')

# Plotting the rewards obtained at each step
rewards = [data['initial_data']['rew']] + \
          [step['rew']['Agent_1'] for step in data['steps_data']]

plt.figure(figsize=(8, 6))
plt.plot(index, rewards, '-o', color='red', markersize=4)
plt.title("Rewards")
plt.xlabel("Step Number")
plt.ylabel("Reward")
plt.show()
