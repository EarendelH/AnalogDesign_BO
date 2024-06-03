# Import required libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Check if the Excel file path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <excel_file_path>")
    sys.exit(1)

# Get the Excel file path from the command-line argument
excel_file_path = sys.argv[1]

# Read the input data from the Excel file
data = pd.read_excel(excel_file_path)

# Separate input and output features
X = data.iloc[:, 22:]  # Input features
y = data.iloc[:, :22]  # Output features

# Create a MultiOutputRegressor instance with RandomForestRegressor as the base estimator
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
multi_output_rf = MultiOutputRegressor(rf)

# Train the multi-output model on the input and output features
multi_output_rf.fit(X, y)

# Get the directory path of the Excel file
excel_dir = os.path.dirname(excel_file_path)

# Print the feature importances for each output
feature_importances = multi_output_rf.estimators_[0].feature_importances_
for i in range(y.shape[1]):
    sorted_importances = sorted(zip(feature_importances[i], X.columns), reverse=True)
    print(f"Feature Importances for Output {i}:")
    for importance, feature in sorted_importances:
        print(f"{feature}: {importance}")
    print()

# Save the feature importances as a table
with open(os.path.join(excel_dir, 'feature_importances.txt'), 'w') as file:
    for i in range(y.shape[1]):
        feature_importances = multi_output_rf.estimators_[i].feature_importances_
        sorted_importances = sorted(zip(feature_importances, X.columns), reverse=True)
        file.write(f"Feature Importances for Output {i}:\n")
        for importance, feature in sorted_importances:
            file.write(f"{feature}: {importance}\n")
        file.write("\n")

# Plot the feature importances for the first output as a bar chart
plt.figure(figsize=(10, 6))
sorted_features = sorted(zip(feature_importances[0], X.columns), reverse=True)
x_pos = range(len(sorted_features))
importances, labels = zip(*sorted_features)
plt.bar(x_pos, importances, align='center')
plt.xticks(x_pos, labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances for Output 0')
plt.tight_layout()
plot_file_path = os.path.join(excel_dir, 'feature_importances_output_0.png')
plt.savefig(plot_file_path)