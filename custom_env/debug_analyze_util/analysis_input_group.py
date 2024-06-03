import cudf
import cuml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Setup command line arguments
parser = argparse.ArgumentParser(description="Run Random Forest to group input features based on their impact on outputs.")
parser.add_argument('--file_path', type=str, help="Path to the input xlsx file")
args = parser.parse_args()

# Read data
data = pd.read_excel(args.file_path)
df = cudf.from_pandas(data)

# Separate output and input data
output_df = df.iloc[:, :22]
input_df = df.iloc[:, 22:]

# Compute feature importance for each output using Random Forest
features_importance = cudf.DataFrame()
for column in output_df.columns:
    X = input_df
    y = output_df[column]
    rf_model = cuml.ensemble.RandomForestRegressor()
    rf_model.fit(X, y)
    features_importance[column] = rf_model.feature_importances_

# Compute average importance across all outputs
average_importance = features_importance.mean(axis=1)

# Group input features into 5 groups based on their impact
labels = pd.cut(average_importance.to_array(), bins=5, labels=np.arange(5))

# Store grouped features in a DataFrame
grouped_input_features = cudf.DataFrame({
    'Feature': input_df.columns,
    'Group': labels
})

# Define the folder where to save files
folder_path = os.path.dirname(args.file_path)

# Save grouped data to CSV
grouped_input_features.to_csv(os.path.join(folder_path, 'feature_groups.csv'))

# Plot the results
plt.bar(grouped_input_features['Feature'].to_array(), grouped_input_features['Group'].to_array())
plt.xlabel('Feature')
plt.ylabel('Group')
plt.title('Feature Importance Grouping')
plt.savefig(os.path.join(folder_path, 'feature_groups.png'))
plt.show()
