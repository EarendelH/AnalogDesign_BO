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
        print("Evaluating different numbers of clusters...")
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
        print(f"Recommended number of clusters (based on silhouette score): {best_n}")

        return results, best_n

    except Exception as e:
        print(f"Error in finding optimal clusters: {str(e)}")
        sys.exit(1)


def find_optimal_dbscan_params(distance_matrix, signals=None, min_samples_range=None, n_jobs=1, output_dir='.'):
    """
    Find optimal parameters for DBSCAN by using k-distance plots
    通过使用k距离图找到DBSCAN的最优参数

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        signals (list, optional): Original signals for additional parameter estimation
                                 用于附加参数估计的原始信号
        min_samples_range (list, optional): Range of min_samples values to try
                                           要尝试的min_samples值范围
        n_jobs (int, optional): Number of parallel jobs
                                并行作业数
        output_dir (str, optional): Output directory path
                                   输出目录路径

    Returns:
        tuple: (optimal_eps, optimal_min_samples, k_distances_plot)
               最优eps，最优min_samples，k距离图
    """
    try:
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import MDS
        from sklearn.neighbors import NearestNeighbors
        from scipy.signal import find_peaks
        from sklearn.cluster import DBSCAN
        from kneed import KneeLocator

        # Auto-determine min_samples based on data dimensionality if not provided
        # 如果未提供，则根据数据维度自动确定min_samples
        if min_samples_range is None:
            if signals is not None:
                # Use signal length as dimensionality
                # 使用信号长度作为维度
                signal_length = len(signals[0])
                data_size = len(signals)

                # Rule of thumb: min_samples around 2x dimensionality but adjusted based on dataset size
                # 经验法则：min_samples约为维度的2倍，但根据数据集大小进行调整
                base_min_samples = max(3, min(signal_length // 50, 10))

                if data_size < 20:
                    # For small datasets, use smaller values
                    # 对于小数据集，使用较小的值
                    min_samples_range = [max(2, base_min_samples - 2), base_min_samples, base_min_samples + 2]
                elif data_size < 100:
                    # For medium datasets
                    # 对于中等数据集
                    min_samples_range = [base_min_samples, base_min_samples + 5, base_min_samples + 10]
                else:
                    # For large datasets
                    # 对于大数据集
                    min_samples_range = [base_min_samples, base_min_samples + 10, base_min_samples + 20]
            else:
                # Default range if signals not provided
                # 如果未提供信号，则使用默认范围
                min_samples_range = [5, 10, 15]

        print(f"Finding optimal DBSCAN parameters using min_samples range: {min_samples_range}")

        # Create directory for k-distance plots
        # 创建k距离图的目录
        kdist_dir = os.path.join(output_dir, 'kdistance_plots')
        os.makedirs(kdist_dir, exist_ok=True)

        # MDS to convert distance matrix to spatial coordinates
        # 使用MDS将距离矩阵转换为空间坐标
        print("Converting distance matrix to spatial coordinates using MDS...")
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=n_jobs)
        pos = mds.fit_transform(distance_matrix)

        results = {}
        k_distance_plots = {}

        # For each min_samples value, find optimal eps
        # 对于每个min_samples值，找到最优eps
        for min_samples in min_samples_range:
            print(f"Analyzing for min_samples={min_samples}...")

            # Calculate k-distance (k = min_samples)
            # 计算k距离（k = min_samples）
            nbrs = NearestNeighbors(n_neighbors=min_samples + 1).fit(pos)
            distances, indices = nbrs.kneighbors(pos)

            # Sort distances to the k-th nearest neighbor
            # 对到第k个最近邻居的距离进行排序
            k_distances = np.sort(distances[:, min_samples])

            # Create k-distance plot
            # 创建k距离图
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(k_distances)), k_distances)
            plt.xlabel('Points sorted by distance')
            plt.ylabel(f'Distance to {min_samples}th nearest neighbor')
            plt.title(f'k-distance plot (k={min_samples})')

            # Try to find the "elbow" point using the kneed package
            # 使用kneed包尝试找到"拐点"
            try:
                kneedle = KneeLocator(
                    range(len(k_distances)),
                    k_distances,
                    S=1.0,
                    curve='convex',
                    direction='increasing'
                )
                elbow_index = kneedle.elbow
                eps = k_distances[elbow_index] if elbow_index is not None else None

                if eps is not None:
                    plt.axvline(x=elbow_index, color='r', linestyle='--', label=f'Elbow at eps={eps:.4f}')
                    plt.axhline(y=eps, color='g', linestyle='--')
                else:
                    # If kneed fails, try a simple method to find a significant change in slope
                    # 如果kneed失败，尝试使用简单方法找到斜率的显著变化
                    diffs = np.diff(k_distances)
                    eps_index = np.argmax(diffs) + 1
                    eps = k_distances[eps_index]
                    plt.axvline(x=eps_index, color='r', linestyle='--', label=f'Suggested eps={eps:.4f}')
                    plt.axhline(y=eps, color='g', linestyle='--')
            except Exception as e:
                print(f"Warning: Could not automatically find elbow point: {str(e)}")
                # Use a percentile-based estimate for eps
                # 使用基于百分位的估计值作为eps
                eps_index = int(len(k_distances) * 0.95)  # 95th percentile
                eps = k_distances[eps_index]
                plt.axvline(x=eps_index, color='r', linestyle='--', label=f'95th percentile eps={eps:.4f}')
                plt.axhline(y=eps, color='g', linestyle='--')

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            # 保存图
            plot_filename = f'kdistance_plot_k{min_samples}.png'
            output_file = os.path.join(kdist_dir, plot_filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"K-distance plot saved as '{output_file}'")

            # Test the clustering with these parameters
            # 使用这些参数测试聚类
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            dbscan.fit(distance_matrix)

            # Get clustering statistics
            # 获取聚类统计
            labels = dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)

            print(f"DBSCAN with eps={eps:.4f}, min_samples={min_samples}:")
            print(f"  Found {n_clusters} clusters and {n_noise} noise points ({noise_ratio:.1%} noise)")

            # Only calculate silhouette score if there are multiple clusters and not all points are noise
            # 只有在有多个簇且不是所有点都是噪声的情况下才计算轮廓系数
            score = -1  # Default for invalid clustering
            if n_clusters > 1 and noise_ratio < 1.0:
                try:
                    from sklearn.metrics import silhouette_score
                    # For silhouette calculation, we need to exclude noise points
                    # 计算轮廓系数时，需要排除噪声点
                    mask = labels != -1
                    if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                        sub_distance = distance_matrix[mask][:, mask]
                        sub_labels = labels[mask]
                        score = silhouette_score(sub_distance, sub_labels, metric='precomputed')
                        print(f"  Silhouette score: {score:.4f}")
                except Exception as e:
                    print(f"  Error calculating silhouette score: {str(e)}")
            else:
                print("  Invalid clustering for silhouette score calculation")

            # Store results
            # 存储结果
            results[(min_samples, eps)] = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette': score
            }

            k_distance_plots[min_samples] = {
                'distances': k_distances,
                'eps': eps,
                'plot_file': output_file
            }

        # Save all results to a file
        # 将所有结果保存到文件
        results_file = os.path.join(output_dir, 'dbscan_parameter_search.txt')
        with open(results_file, 'w') as f:
            f.write("===== DBSCAN Parameter Search Results =====\n\n")
            f.write(f"Tested min_samples values: {min_samples_range}\n\n")

            for (min_samples, eps), res in results.items():
                f.write(f"Parameters: min_samples={min_samples}, eps={eps:.4f}\n")
                f.write(f"  Number of clusters: {res['n_clusters']}\n")
                f.write(f"  Number of noise points: {res['n_noise']} ({res['noise_ratio']:.1%})\n")
                f.write(f"  Silhouette score: {res['silhouette']:.4f}\n\n")

        print(f"Parameter search results saved to '{results_file}'")

        # Find optimal parameters based on silhouette score and reasonable noise ratio
        # 基于轮廓系数和合理的噪声比例找到最优参数
        valid_params = {
            params: res for params, res in results.items()
            if res['n_clusters'] > 1 and res['noise_ratio'] < 0.5 and res['silhouette'] > -1
        }

        if valid_params:
            # Find parameters with highest silhouette score
            # 找到具有最高轮廓系数的参数
            optimal_params = max(valid_params.items(), key=lambda x: x[1]['silhouette'])[0]
            optimal_min_samples, optimal_eps = optimal_params

            print(f"\nRecommended DBSCAN parameters:")
            print(f"  eps={optimal_eps:.4f}")
            print(f"  min_samples={optimal_min_samples}")
            print(f"  Expected result: {valid_params[optimal_params]['n_clusters']} clusters, "
                  f"{valid_params[optimal_params]['noise_ratio']:.1%} noise ratio, "
                  f"silhouette={valid_params[optimal_params]['silhouette']:.4f}")

            return optimal_eps, optimal_min_samples, k_distance_plots
        else:
            print("\nCould not find optimal parameters with good clustering quality.")
            print("You may need to try different parameter ranges or a different clustering algorithm.")

            # Return the parameters with the most reasonable clustering
            # 返回具有最合理聚类的参数
            reasonable_params = {
                params: res for params, res in results.items()
                if res['n_clusters'] > 0 and res['noise_ratio'] < 0.9
            }

            if reasonable_params:
                suggestion = min(reasonable_params.items(), key=lambda x: x[1]['noise_ratio'])[0]
                min_samples_sugg, eps_sugg = suggestion

                print(f"\nSuggested parameters (minimize noise):")
                print(f"  eps={eps_sugg:.4f}")
                print(f"  min_samples={min_samples_sugg}")

                return eps_sugg, min_samples_sugg, k_distance_plots
            else:
                # If all results are poor, return the parameters for the middle min_samples value
                # 如果所有结果都很差，则返回中间min_samples值的参数
                mid_min_samples = min_samples_range[len(min_samples_range) // 2]
                for params, res in results.items():
                    if params[0] == mid_min_samples:
                        return params[1], params[0], k_distance_plots

                # Fallback to first result
                # 回退到第一个结果
                first_params = list(results.keys())[0]
                return first_params[1], first_params[0], k_distance_plots

    except Exception as e:
        print(f"Error in finding optimal DBSCAN parameters: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return default values
        # 返回默认值
        return 0.5, 5, {}