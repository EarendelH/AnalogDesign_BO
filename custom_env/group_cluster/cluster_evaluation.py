import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import sys


def evaluate_clustering(distance_matrix, labels):
    """
    Evaluate clustering results using silhouette score
    使用轮廓系数评估聚类结果

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        labels (numpy.ndarray): Cluster labels
                               簇标签

    Returns:
        float: Silhouette score
               轮廓系数
    """
    try:
        # Check if all points are assigned to the same cluster
        # 检查是否所有点都分配给了同一个簇
        if len(set(labels)) <= 1:
            return -1  # Invalid clustering for evaluation

        # Calculate silhouette score
        # 计算轮廓系数
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
        return score

    except Exception as e:
        print(f"Error in clustering evaluation: {str(e)}")
        return -1


def find_optimal_clusters(distance_matrix, signals, max_clusters=10, linkage='average'):
    """
    Find optimal number of clusters using multiple metrics
    使用多个指标找到最优的簇数

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        signals (list): List of signal arrays
                       信号数组列表
        max_clusters (int, optional): Maximum number of clusters to try
                                     尝试的最大簇数
        linkage (str, optional): Linkage criterion for hierarchical clustering
                                层次聚类的连接方法

    Returns:
        dict: Dictionary with scores for different numbers of clusters
              包含不同簇数得分的字典
    """
    try:
        # Convert signals to array format for calinski_harabasz_score
        # 将信号转换为数组格式，用于calinski_harabasz_score
        signals_2d = np.array([signal.flatten() for signal in signals])

        # Initialize results dictionary
        # 初始化结果字典
        results = {
            'n_clusters': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }

        # Try different numbers of clusters
        # 尝试不同数量的簇
        print(f"Evaluating different numbers of clusters with {linkage} linkage...")
        for n_clusters in range(2, min(max_clusters + 1, len(signals))):
            print(f"  Testing {n_clusters} clusters...")

            # Perform hierarchical clustering
            # 执行层次聚类
            from clustering import hierarchical_clustering
            model = hierarchical_clustering(distance_matrix, n_clusters=n_clusters, linkage=linkage)
            labels = model.labels_

            # Calculate silhouette score
            # 计算轮廓系数
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')

            # Calculate Davies-Bouldin index (lower is better)
            # 计算Davies-Bouldin指数（越低越好）
            db_score = davies_bouldin_score(signals_2d, labels)

            # Calculate Calinski-Harabasz index (higher is better)
            # 计算Calinski-Harabasz指数（越高越好）
            ch_score = calinski_harabasz_score(signals_2d, labels)

            # Store results
            # 存储结果
            results['n_clusters'].append(n_clusters)
            results['silhouette'].append(sil_score)
            results['davies_bouldin'].append(db_score)
            results['calinski_harabasz'].append(ch_score)

            print(f"    Silhouette Score: {sil_score:.4f}")
            print(f"    Davies-Bouldin Index: {db_score:.4f}")
            print(f"    Calinski-Harabasz Index: {ch_score:.4f}")

        # Determine recommended number of clusters (based on silhouette score)
        # 确定推荐的簇数（基于轮廓系数）
        best_n = results['n_clusters'][np.argmax(results['silhouette'])]
        print(f"Recommended number of clusters for {linkage} linkage (based on silhouette score): {best_n}")

        return results, best_n

    except Exception as e:
        print(f"Error in finding optimal clusters: {str(e)}")
        sys.exit(1)


def evaluate_all_linkage_methods(distance_matrix, signals, max_clusters=10):
    """
    Evaluate all linkage methods and find optimal number of clusters for each
    评估所有连接方法并找到每种方法的最优簇数

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        signals (list): List of signal arrays
                       信号数组列表
        max_clusters (int, optional): Maximum number of clusters to try
                                     尝试的最大簇数

    Returns:
        tuple: (all_results, best_linkage, best_n_clusters)
               (所有结果, 最佳连接方法, 最佳簇数)
    """
    try:
        # Dictionary to store results for each linkage method
        # 存储每种连接方法结果的字典
        all_results = {}

        # Available linkage methods (excluding 'ward' which doesn't work with precomputed distances)
        # 可用的连接方法（不包括'ward'，因为它不适用于预计算的距离）
        linkage_methods = ['single', 'complete', 'average']

        # For each linkage method, find optimal clusters
        # 对于每种连接方法，找到最优簇数
        for linkage in linkage_methods:
            print(f"\n===== Evaluating {linkage} linkage method =====")
            results, best_n = find_optimal_clusters(distance_matrix, signals, max_clusters, linkage)
            all_results[linkage] = results

        return all_results

    except Exception as e:
        print(f"Error in evaluating all linkage methods: {str(e)}")
        sys.exit(1)