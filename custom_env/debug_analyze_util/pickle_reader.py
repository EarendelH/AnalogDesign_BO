import pickle
import numpy as np
import matplotlib.pyplot as plt

pickle_file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/analysis_file/202401222314028359.pkl"

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

ideal_specs = data['initial_data']['ideal_specs']
ideal_specs_extracted = {key: value['value'] for key, value in ideal_specs.items()}

sim_results = [data['initial_data']['sim_result']['Stability']] + \
              [step['sim_result']['Stability'] for step in data['steps_data']]
index = np.arange(len(sim_results))
phase_margins = [result['phaseMargin'] for result in sim_results]
gain_bandwidths = [result['gainBandWidth'] for result in sim_results]

rewards = [data['initial_data']['rew']] + \
          [step['rew']['Agent_1'] for step in data['steps_data']]

plt.figure(figsize=(8, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(index, phase_margins, '-o', label='Phase Margin', color='blue', markersize=4)
ax1.set_xlabel('Step Number')
ax1.set_ylabel('Phase Margin', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.plot(index, gain_bandwidths, '-o', label='Gain Bandwidth', color='orange', markersize=4)
ax2.set_ylabel('Gain Bandwidth', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.title("Stability Metrics\n" +
          "\n".join([f"{key}: {value}" for key, value in ideal_specs_extracted.items()]))
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(index, rewards, '-o', color='red', markersize=4)
plt.title("Rewards\n" + "\n".join([f"{key}: {value}" for key, value in ideal_specs_extracted.items()]))
plt.xlabel("Step Number")
plt.ylabel("Reward")
plt.show()
