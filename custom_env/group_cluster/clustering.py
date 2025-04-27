import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from tslearn.clustering import TimeSeriesKMeans
import sys


def hierarchical_clustering(distance_matrix, n_clusters=None, linkage='ward'):
    """
    Perform hierarchical clustering based on the DTW distance matrix
    基于DTW距离矩阵进行层次聚类

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        n_clusters (int, optional): Number of clusters. If None, no clusters are formed
                                    簇的数量。如果为None，则不形成簇
        linkage (str, optional): Linkage criterion ('ward', 'complete', 'average', 'single')
                                 连接标准（'ward'、'complete'、'average'、'single'）

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted clustering model
                                                拟合的聚类模型
    """
    try:
        print(f"Performing hierarchical clustering with {linkage} linkage...")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',  # 使用'metric'替代'affinity'
            linkage=linkage
        )

        # Fit the model to get cluster labels
        # 拟合模型以获取簇标签
        model.fit(distance_matrix)

        return model

    except Exception as e:
        print(f"Error in hierarchical clustering: {str(e)}")
        sys.exit(1)


def time_series_kmeans(signals, n_clusters, metric="dtw", random_state=42):
    """
    Perform TimeSeriesKMeans clustering directly on the signals
    直接在信号上执行TimeSeriesKMeans聚类

    Args:
        signals (list): List of signal arrays
                        信号数组列表
        n_clusters (int): Number of clusters
                          簇的数量
        metric (str, optional): Distance metric ("dtw", "softdtw")
                                距离度量（"dtw"，"softdtw"）
        random_state (int, optional): Random state for reproducibility
                                      用于再现性的随机状态

    Returns:
        tslearn.clustering.TimeSeriesKMeans: Fitted clustering model
                                            拟合的聚类模型
    """
    try:
        # Convert list of signals to 3D array required by tslearn
        # 将信号列表转换为tslearn所需的3D数组
        signals_array = np.array([s.reshape(-1, 1) for s in signals])

        print(f"Performing TimeSeriesKMeans with {n_clusters} clusters...")
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            random_state=random_state,
            verbose=True
        )

        # Fit the model to get cluster labels and centroids
        # 拟合模型以获取簇标签和质心
        model.fit(signals_array)

        return model

    except Exception as e:
        print(f"Error in TimeSeriesKMeans: {str(e)}")
        sys.exit(1)


def dbscan_clustering(distance_matrix, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering based on the DTW distance matrix
    基于DTW距离矩阵进行DBSCAN聚类

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        eps (float, optional): Maximum distance between samples for one to be considered as
                              in the neighborhood of the other
                              样本之间的最大距离，使一个样本被视为另一个样本的邻居
        min_samples (int, optional): Minimum number of samples in a neighborhood for a point
                                    to be considered as a core point
                                    一个点被视为核心点的邻域中的最小样本数

    Returns:
        sklearn.cluster.DBSCAN: Fitted clustering model
                               拟合的聚类模型
    """
    try:
        print(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        )

        # Fit the model to get cluster labels
        # 拟合模型以获取簇标签
        model.fit(distance_matrix)

        # Print number of clusters found and noise points
        # 打印找到的簇的数量和噪声点
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        n_noise = list(model.labels_).count(-1)
        print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")

        return model

    except Exception as e:
        print(f"Error in DBSCAN clustering: {str(e)}")
        sys.exit(1)