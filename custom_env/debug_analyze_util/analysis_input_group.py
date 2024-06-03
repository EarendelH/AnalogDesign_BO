import pandas as pd
import numpy as np
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process Excel file and store results")
parser.add_argument('file_path', type=str, help='Path to the input Excel file')
args = parser.parse_args()

# Read Excel file
file_path = args.file_path
data = pd.read_excel(file_path)

# Separate input and output data
output_data = data.iloc[:, :21]
input_data = data.iloc[:, 21:]

# Convert data to numpy arrays
X = input_data.to_numpy()
Y = output_data.to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Build the GPU-accelerated Random Forest model
rf = cuRF(n_estimators=100, random_state=0)
rf.fit(X_train, Y_train)

# Calculate feature importance
feature_importances = rf.feature_importances_

# Convert feature importance to a pandas DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': input_data.columns,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Divide into five groups
group_size = len(feature_importance_df) // 5
groups = {}
for i in range(5):
    start_index = i * group_size
    if i == 4:  # Ensure the last group includes all remaining features
        groups[f'Group {i+1}'] = feature_importance_df.iloc[start_index:]
    else:
        groups[f'Group {i+1}'] = feature_importance_df.iloc[start_index:start_index + group_size]

# Save grouped results to Excel file
output_dir = os.path.dirname(file_path)
output_excel_path = os.path.join(output_dir, 'Feature_Importance_Groups.xlsx')
with pd.ExcelWriter(output_excel_path) as writer:
    for group_name, group_data in groups.items():
        group_data.to_excel(writer, sheet_name=group_name, index=False)

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Annotate each bar with the feature name
for index, value in enumerate(feature_importance_df['Importance']):
    plt.text(value, index, f'{value:.2f}')

# Save the plot
output_plot_path = os.path.join(output_dir, 'Feature_Importance_Plot.png')
plt.savefig(output_plot_path)
plt.show()
