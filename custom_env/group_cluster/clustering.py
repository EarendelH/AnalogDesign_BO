import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sys


def hierarchical_clustering(distance_matrix, n_clusters=None, linkage='ward', distance_threshold=None):
    """
    Perform hierarchical clustering based on the DTW distance matrix
    基于DTW距离矩阵进行层次聚类

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        n_clusters (int, optional): Number of clusters. If None, distance_threshold must be provided
                                    簇的数量。如果为None，则必须提供distance_threshold
        linkage (str, optional): Linkage criterion ('ward', 'complete', 'average', 'single')
                                 连接标准（'ward'、'complete'、'average'、'single'）
        distance_threshold (float, optional): The distance threshold to stop merging clusters
                                             停止合并簇的距离阈值

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted clustering model
                                                拟合的聚类模型
    """
    try:
        # Check if ward linkage is being used with precomputed distances (incompatible)
        # 检查是否正在使用ward连接方法与预计算距离（不兼容）
        if linkage == 'ward':
            print("Warning: Ward linkage cannot be used with precomputed distances.")
            print("Automatically switching to 'average' linkage...")
            linkage = 'average'

        # Ensure exactly one of n_clusters and distance_threshold is set
        # 确保n_clusters和distance_threshold中只有一个被设置
        if n_clusters is None and distance_threshold is None:
            distance_threshold = 0.0  # This will create a separate cluster for each sample
            print(f"Using distance_threshold={distance_threshold} for dendrogram creation")
        elif n_clusters is not None and distance_threshold is not None:
            print(
                f"Both n_clusters and distance_threshold were provided. Using n_clusters={n_clusters} and ignoring distance_threshold.")
            distance_threshold = None

        print(f"Performing hierarchical clustering with {linkage} linkage...")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage=linkage,
            distance_threshold=distance_threshold
        )

        # Fit the model to get cluster labels
        # 拟合模型以获取簇标签
        model.fit(distance_matrix)

        return model

    except Exception as e:
        print(f"Error in hierarchical clustering: {str(e)}")
        sys.exit(1)