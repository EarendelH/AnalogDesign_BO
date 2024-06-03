import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def convert_to_parquet(excel_file_path):
    parquet_file_path = excel_file_path.replace('.xlsx', '.parquet')
    data = pd.read_excel(excel_file_path)
    data.to_parquet(parquet_file_path)
    return parquet_file_path

def train_model(X, y):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    multi_output_rf = MultiOutputRegressor(rf)
    multi_output_rf.fit(X, y)
    return multi_output_rf

def compute_feature_importances(estimator):
    return estimator.feature_importances_

if len(sys.argv) < 2:
    print("Usage: python script.py <excel_file_path>")
    sys.exit(1)

excel_file_path = sys.argv[1]

print("Converting Excel file to Parquet format...")
parquet_file_path = convert_to_parquet(excel_file_path)
print("Conversion complete.")

print("Reading Parquet file...")
data = pd.read_parquet(parquet_file_path)
print("Data loaded.")

X = data.iloc[:, 22:]
y = data.iloc[:, :22]

print("Training model...")
multi_output_rf = train_model(X, y)
print("Model training complete.")

print("Computing feature importances...")
feature_importances = Parallel(n_jobs=-1)(
    delayed(compute_feature_importances)(estimator) for estimator in tqdm(multi_output_rf.estimators_, desc="Computing feature importances")
)
print("Feature importance computation complete.")

overall_feature_importances = np.mean(feature_importances, axis=0)

sorted_features = sorted(zip(overall_feature_importances, X.columns), reverse=True)

groups = []
current_group = []

for importance, feature in sorted_features:
    if current_group and abs(importance - current_group[0][0]) > 0.02:
        groups.append(current_group)
        current_group = []
    current_group.append((importance, feature))

if current_group:
    groups.append(current_group)

groups = [[feature for _, feature in group] for group in groups]

print("Feature Grouping:")
for i, group in enumerate(groups):
    print(f"Group {i+1}: {', '.join(group)}")

excel_dir = os.path.dirname(excel_file_path)

grouping_file_path = os.path.join(excel_dir, 'feature_grouping.txt')
with open(grouping_file_path, 'w') as file:
    file.write("Feature Grouping:\n\n")
    for i, group in enumerate(groups):
        file.write(f"Group {i+1}: {', '.join(group)}\n")

plt.figure(figsize=(10, 6))
x_pos = range(len(sorted_features))
importances, labels = zip(*sorted_features)
plt.bar(x_pos, importances, align='center')
plt.xticks(x_pos, labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Overall Feature Importances')
plt.tight_layout()
plot_file_path = os.path.join(excel_dir, 'overall_feature_importances.png')
plt.savefig(plot_file_path)