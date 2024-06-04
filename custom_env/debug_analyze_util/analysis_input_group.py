import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
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


def plot_feature_importances(importances, output_name, excel_dir):
    """
    Plot feature importances for a specific output and save the plot.

    Args:
        importances (numpy.ndarray): Feature importances.
        output_name (str): Name of the output feature.
        excel_dir (str): Directory path of the Excel file.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances, align='center')
    plt.xticks(range(len(importances)), X.columns, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances for Output: {output_name}')
    plt.tight_layout()
    plot_file_path = os.path.join(excel_dir, f'feature_importances_{output_name}.png')
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

# Plot feature importances for each output and save the plots
for i in range(y.shape[1]):
    plot_feature_importances(feature_importances_tensor[i], y.columns[i], excel_dir)

# Normalize feature importances for each output
normalized_importances = feature_importances_tensor / feature_importances_tensor.sum(axis=1, keepdims=True)

# Compute feature similarity matrix
feature_similarity_matrix = np.zeros((X.shape[1], X.shape[1]))
for i in range(y.shape[1]):
    output_importances = normalized_importances[i]
    feature_similarity_matrix += np.outer(output_importances, output_importances)

# Save feature similarity matrix to file
similarity_matrix_file_path = os.path.join(excel_dir, 'feature_similarity_matrix.txt')
np.savetxt(similarity_matrix_file_path, feature_similarity_matrix, fmt='%.4f')

# Perform hierarchical clustering on the feature similarity matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(feature_similarity_matrix, method='ward')

# Visualize the hierarchical clustering results
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=list(X.columns), orientation='right')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering')
plt.tight_layout()

hierarchical_plot_file_path = os.path.join(excel_dir, 'hierarchical_clustering.png')
plt.savefig(hierarchical_plot_file_path)

# Group features based on the hierarchical clustering results
max_distance = 0.5
hierarchical_clusters = fcluster(Z, max_distance, criterion='distance')

hierarchical_feature_groups = {}
for feature, cluster in zip(X.columns, hierarchical_clusters):
    if cluster not in hierarchical_feature_groups:
        hierarchical_feature_groups[cluster] = []
    hierarchical_feature_groups[cluster].append(feature)

# Perform k-means clustering on the feature similarity matrix
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(feature_similarity_matrix)
kmeans_clusters = kmeans.labels_

kmeans_feature_groups = {}
for feature, cluster in zip(X.columns, kmeans_clusters):
    if cluster not in kmeans_feature_groups:
        kmeans_feature_groups[cluster] = []
    kmeans_feature_groups[cluster].append(feature)

# Save clustering results to file
clustering_results_file_path = os.path.join(excel_dir, 'clustering_results.txt')
with open(clustering_results_file_path, 'w') as file:
    file.write("Hierarchical Clustering Results:\n\n")
    for cluster, features in hierarchical_feature_groups.items():
        file.write(f"Cluster {cluster}: {', '.join(features)}\n")

    file.write("\nK-means Clustering Results:\n\n")
    for cluster, features in kmeans_feature_groups.items():
        file.write(f"Cluster {cluster}: {', '.join(features)}\n")

# Print clustering results to console
print("Hierarchical Clustering Results:")
for cluster, features in hierarchical_feature_groups.items():
    print(f"Cluster {cluster}: {', '.join(features)}")
print("\nK-means Clustering Results:")
for cluster, features in kmeans_feature_groups.items():
    print(f"Cluster {cluster}: {', '.join(features)}")
