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

# Create an instance of the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Train the model on the input and output features
rf.fit(X, y)

# Get the feature importances
feature_importances = rf.feature_importances_

# Sort the feature importances in descending order
sorted_importances = sorted(zip(feature_importances, X.columns), reverse=True)

# Initialize groups and temporary list
groups = []
current_group = []

# Iterate over sorted feature importances
for importance, feature in sorted_importances:
    # If the current group has a large importance gap, start a new group
    if current_group and abs(importance - current_group[0][0]) > 0.05:
        groups.append(current_group)
        current_group = []
    current_group.append((importance, feature))

# Add the last group
if current_group:
    groups.append(current_group)

# Extract feature names from groups
groups = [[feature for _, feature in group] for group in groups]

# Get the directory path of the Excel file
excel_dir = os.path.dirname(excel_file_path)

# Print the feature grouping
print("Feature Grouping:")
for i, group in enumerate(groups):
    print(f"Group {i+1}: {', '.join(group)}")

# Save the feature grouping as a table
grouping_file_path = os.path.join(excel_dir, 'feature_grouping.txt')
with open(grouping_file_path, 'w') as file:
    file.write("Feature Grouping:\n\n")
    for i, group in enumerate(groups):
        file.write(f"Group {i+1}: {', '.join(group)}\n")

# Plot the feature importances as a bar chart
plt.figure(figsize=(10, 6))
sorted_features = sorted(zip(feature_importances, X.columns), reverse=True)
x_pos = range(len(sorted_features))
importances, labels = zip(*sorted_features)
plt.bar(x_pos, importances, align='center')
plt.xticks(x_pos, labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plot_file_path = os.path.join(excel_dir, 'feature_importances.png')
plt.savefig(plot_file_path)
