# Import required libraries
import pandas as pd
import cuml
from cuml.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import sys

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

# Create a list to store the trained models
models = []

# Train a separate model for each output feature
for i in range(y.shape[1]):
    output_target = y.iloc[:, i]
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, output_target)
    models.append(rf)

# Get the directory path of the Excel file
excel_dir = os.path.dirname(excel_file_path)

# Print the feature importances for each model
for i, model in enumerate(models):
    feature_importances = model.feature_importances_
    sorted_importances = sorted(zip(feature_importances, X.columns), reverse=True)
    print(f"Feature Importances for Output {i}:")
    for importance, feature in sorted_importances:
        print(f"{feature}: {importance}")
    print()

# Save the feature importances as a table
with open(os.path.join(excel_dir, 'feature_importances.txt'), 'w') as file:
    for i, model in enumerate(models):
        feature_importances = model.feature_importances_
        sorted_importances = sorted(zip(feature_importances, X.columns), reverse=True)
        file.write(f"Feature Importances for Output {i}:\n")
        for importance, feature in sorted_importances:
            file.write(f"{feature}: {importance}\n")
        file.write("\n")
