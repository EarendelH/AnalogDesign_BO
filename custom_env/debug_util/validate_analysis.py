import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

pickle_dir = "/Users/hanwu/Downloads/Pickle/"
pickle_files = os.listdir(pickle_dir)

ideal_space_list = []
step_num_list = []

for pickle_file in pickle_files:
    if pickle_file.endswith('.pkl'):
        with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
            try:
                data = pickle.load(f)
            except pickle.UnpicklingError:
                print(f"Cannot unpickle file: {pickle_file}")
                continue

        ideal_specs = data['initial_data']['ideal_specs']
        ideal_specs_extracted = {key: value['value'] for key, value in ideal_specs.items()}
        ideal_space_list.append(list(ideal_specs_extracted.values()))

        step_num_list.append(len(data['steps_data']))


print(f"Ideal Space: {ideal_space_list} with length {len(ideal_space_list)}")
print(f"Searched Steps: {step_num_list} with length {len(step_num_list)}")

# Calculate mean and variance of step_num
step_num_list = np.array(step_num_list)
print(f"Mean of step_num: {np.mean(step_num_list)}")
print(f"Variance of step_num: {np.var(step_num_list)}")

GBW_list = np.array([item[0] for item in ideal_space_list])
PM_list = np.array([item[1] for item in ideal_space_list])
step_num_list = np.array(step_num_list)

# Plot ideal_spac distribution in scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(GBW_list, PM_list)
plt.xlabel("GBW")
plt.ylabel("PM")
plt.title("Ideal Space Distribution")
plt.show()

# Plot step_num and ideal_space distribution in 3D scatterplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PM_list, GBW_list, step_num_list)
ax.set_xlabel("PM")
ax.set_ylabel("GBW")
ax.set_zlabel("Step Number")
plt.show()

# Plot step_num and ideal_space distribution in 3D scatterplot, filtering out step_num > 100
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PM_list, GBW_list, step_num_list)
ax.set_xlabel("PM")
ax.set_ylabel("GBW")
ax.set_zlabel("Step Number")
ax.set_zlim(0, 50)
plt.show()

# # Create a 2D histogram with the x, y, and weights (z-values).
# plt.figure(figsize=(8, 6))
# plt.hist2d(GBW_list, PM_list, weights=step_num_list, bins=[50, 50], cmap='plasma')
#
# # Add a colorbar to show the scale of step numbers.
# plt.colorbar()
# # Set labels for the axes.
# plt.xlabel("GBW")
# plt.ylabel("PM")
#
# # Show the heatmap.
# plt.show()