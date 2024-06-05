import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
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


def compute_impurity_importances(model):
    """
    Compute feature importances for a given model using the impurity-based method.

    Args:
        model (RandomForestRegressor): Trained random forest model.

    Returns:
        numpy.ndarray: Feature importances from the impurity-based method.
    """
    return model.feature_importances_

def compute_permutation_importances(model, X, y):
    """
    Compute feature importances for a given model using the permutation-based method.

    Args:
        model (RandomForestRegressor): Trained random forest model.
        X (pandas.DataFrame): Input features.
        y (pandas.Series): Output feature.

    Returns:
        numpy.ndarray: Feature importances from the permutation-based method.
    """
    y_array = y.to_numpy().copy()
    return permutation_importance(model, X, y_array, n_repeats=10, random_state=42).importances_mean



def plot_feature_importances(importances, output_name, excel_dir, method):
    """
    Plot feature importances for a specific output and save the plot.

    Args:
        importances (numpy.ndarray): Feature importances.
        output_name (str): Name of the output feature.
        excel_dir (str): Directory path of the Excel file.
        method (str): Method used for calculating feature importances ('impurity' or 'permutation').
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances, align='center')
    plt.xticks(range(len(importances)), X.columns, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances for Output: {output_name} ({method})')
    plt.tight_layout()
    plot_file_path = os.path.join(excel_dir, f'feature_importances_{output_name}_{method}.png')
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
impurity_importances_tensor = np.array(Parallel(n_jobs=-1)(
    delayed(compute_impurity_importances)(model) for model in tqdm(models, desc="Computing impurity-based feature importances")
))
permutation_importances_tensor = np.array(Parallel(n_jobs=-1)(
    delayed(compute_permutation_importances)(model, X, y.iloc[:, i]) for i, model in enumerate(tqdm(models, desc="Computing permutation-based feature importances"))
))
print("Feature importance computation complete.")

print("Impurity-based Feature Importances Tensor:")
print(impurity_importances_tensor)
print("Permutation-based Feature Importances Tensor:")
print(permutation_importances_tensor)

excel_dir = os.path.dirname(excel_file_path)

for method, importances_tensor in [('impurity', impurity_importances_tensor), ('permutation', permutation_importances_tensor)]:
    tensor_file_path = os.path.join(excel_dir, f'feature_importances_tensor_{method}.txt')
    np.savetxt(tensor_file_path, importances_tensor, fmt='%.4f')

    for i in range(y.shape[1]):
        plot_feature_importances(importances_tensor[i], y.columns[i], excel_dir, method)

    normalized_importances = importances_tensor / importances_tensor.sum(axis=1, keepdims=True)

    feature_similarity_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(y.shape[1]):
        output_importances = normalized_importances[i]
        feature_similarity_matrix += np.outer(output_importances, output_importances)

    similarity_matrix_file_path = os.path.join(excel_dir, f'feature_similarity_matrix_{method}.txt')
    np.savetxt(similarity_matrix_file_path, feature_similarity_matrix, fmt='%.4f')

    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    Z = linkage(feature_similarity_matrix, method='ward')

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=list(X.columns), orientation='right')
    plt.xlabel('Features')
    plt.ylabel('Distance')
    plt.title(f'Hierarchical Clustering ({method})')
    plt.tight_layout()

    hierarchical_plot_file_path = os.path.join(excel_dir, f'hierarchical_clustering_{method}.png')
    plt.savefig(hierarchical_plot_file_path)

    max_distance = 0.5
    hierarchical_clusters = fcluster(Z, max_distance, criterion='distance')

    hierarchical_feature_groups = {}
    for feature, cluster in zip(X.columns, hierarchical_clusters):
        if cluster not in hierarchical_feature_groups:
            hierarchical_feature_groups[cluster] = []
        hierarchical_feature_groups[cluster].append(feature)

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_similarity_matrix)
    kmeans_clusters = kmeans.labels_

    kmeans_feature_groups = {}
    for feature, cluster in zip(X.columns, kmeans_clusters):
        if cluster not in kmeans_feature_groups:
            kmeans_feature_groups[cluster] = []
        kmeans_feature_groups[cluster].append(feature)

    clustering_results_file_path = os.path.join(excel_dir, f'clustering_results_{method}.txt')
    with open(clustering_results_file_path, 'w') as file:
        file.write(f"Hierarchical Clustering Results ({method}):\n\n")
        for cluster, features in hierarchical_feature_groups.items():
            file.write(f"Cluster {cluster}: {', '.join(features)}\n")

        file.write(f"\nK-means Clustering Results ({method}):\n\n")
        for cluster, features in kmeans_feature_groups.items():
            file.write(f"Cluster {cluster}: {', '.join(features)}\n")

    print(f"Hierarchical Clustering Results ({method}):")
    for cluster, features in hierarchical_feature_groups.items():
        print(f"Cluster {cluster}: {', '.join(features)}")
    print(f"\nK-means Clustering Results ({method}):")
    for cluster, features in kmeans_feature_groups.items():
        print(f"Cluster {cluster}: {', '.join(features)}")