import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def convert_to_parquet(excel_file_path):
    """
    Convert an Excel file to Parquet format.

    Args:
        excel_file_path (str): Path to the Excel file.

    Returns:
        str: Path to the converted Parquet file.
    """
    parquet_file_path = excel_file_path.replace('.xlsx', '.parquet')
    data = pd.read_excel(excel_file_path)
    data.to_parquet(parquet_file_path)
    return parquet_file_path

def train_models(X, y):
    """
    Train random forest models for each output feature.

    Args:
        X (pandas.DataFrame): Input features.
        y (pandas.DataFrame): Output features.

    Returns:
        list: Trained random forest models.
    """
    models = []
    for i in range(y.shape[1]):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X, y.iloc[:, i])
        models.append(rf)
    return models

def compute_feature_importances(model):
    """
    Compute feature importances for a given model.

    Args:
        model (RandomForestRegressor): Trained random forest model.

    Returns:
        numpy.ndarray: Feature importances.
    """
    return model.feature_importances_

def plot_feature_importances(importances, output_idx, output_name, excel_dir):
    """
    Plot feature importances for a specific output.

    Args:
        importances (numpy.ndarray): Feature importances.
        output_idx (int): Index of the output feature.
        output_name (str): Name of the output feature.
        excel_dir (str): Directory path of the Excel file.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances, align='center')
    plt.xticks(range(len(importances)), X.columns, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances for Output {output_idx+1}: {output_name}')
    plt.tight_layout()
    plot_file_path = os.path.join(excel_dir, f'feature_importances_output_{output_idx+1}.png')
    plt.savefig(plot_file_path)
    plt.close()

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

X = data.iloc[:, 21:]
y = data.iloc[:, :21]

print("Training models...")
models = train_models(X, y)
print("Model training complete.")

print("Computing feature importances...")
feature_importances_tensor = np.array(Parallel(n_jobs=-1)(
    delayed(compute_feature_importances)(model) for model in tqdm(models, desc="Computing feature importances")
))
print("Feature importance computation complete.")

print("Feature Importances Tensor:")
print(feature_importances_tensor)

excel_dir = os.path.dirname(excel_file_path)
tensor_file_path = os.path.join(excel_dir, 'feature_importances_tensor.txt')
np.savetxt(tensor_file_path, feature_importances_tensor, fmt='%.4f')

# Plot feature importances for each output
for i in range(y.shape[1]):
    plot_feature_importances(feature_importances_tensor[i], i, y.columns[i], excel_dir)

# Normalize feature importances tensor
normalized_importances = feature_importances_tensor / feature_importances_tensor.sum(axis=1, keepdims=True)

# Compute similarity matrix between features
similarity_matrix = np.dot(normalized_importances.T, normalized_importances)

# Perform hierarchical clustering on the similarity matrix
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(similarity_matrix, method='ward')

# Visualize the clustering results
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=list(X.columns), orientation='right')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.title('Feature Clustering')
plt.tight_layout()

plot_file_path = os.path.join(excel_dir, 'feature_clustering.png')
plt.savefig(plot_file_path)

# Group features based on the clustering results
from scipy.cluster.hierarchy import fcluster
max_distance = 0.2
clusters = fcluster(Z, max_distance, criterion='distance')

feature_groups = {}
for feature, cluster in zip(X.columns, clusters):
    if cluster not in feature_groups:
        feature_groups[cluster] = []
    feature_groups[cluster].append(feature)

print("Feature Grouping:")
for cluster, features in feature_groups.items():
    print(f"Group {cluster}: {', '.join(features)}")

grouping_file_path = os.path.join(excel_dir, 'feature_grouping.txt')
with open(grouping_file_path, 'w') as file:
    file.write("Feature Grouping:\n\n")
    for cluster, features in feature_groups.items():
        file.write(f"Group {cluster}: {', '.join(features)}\n")
