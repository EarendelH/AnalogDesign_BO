import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os
import sys

def plot_dendrogram(model, distance_matrix=None, signal_names=None, figsize=(12, 8), output_dir='output', linkage='average'):
    """
    Plot dendrogram from hierarchical clustering
    绘制层次聚类的树状图

    Args:
        model: Fitted AgglomerativeClustering model
               拟合的AgglomerativeClustering模型
        distance_matrix (numpy.ndarray, optional): DTW distance matrix, needed for dendrogram
                                                  DTW距离矩阵，绘制树状图所需
        signal_names (list, optional): List of signal names
                                      信号名称列表
        figsize (tuple, optional): Figure size
                                  图形大小
        output_dir (str, optional): Output directory
                                   输出目录
        linkage (str, optional): Linkage method used
                                使用的连接方法
    """
    try:
        # Use scipy's hierarchical clustering to compute the linkage matrix
        # 使用scipy的层次聚类计算连接矩阵
        if distance_matrix is None:
            print("Error: Distance matrix is required for plotting dendrogram.")
            return

        from scipy.cluster.hierarchy import linkage as scipy_linkage
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        from scipy.spatial.distance import squareform

        # Convert to condensed distance matrix if needed
        # 如需要，将距离矩阵转换为压缩形式
        if distance_matrix.shape[0] == distance_matrix.shape[1]:
            condensed_dist = squareform(distance_matrix)
        else:
            condensed_dist = distance_matrix

        # Compute linkage matrix using the same linkage method as the model
        # 使用与模型相同的连接方法计算连接矩阵
        linkage_matrix = scipy_linkage(condensed_dist, method=model.linkage)

        # Plot the dendrogram
        # 绘制树状图
        plt.figure(figsize=figsize)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage} linkage)', fontsize=15)
        plt.xlabel('Signal Index or Signal Name', fontsize=12)
        plt.ylabel('Distance', fontsize=12)

        labels = signal_names if signal_names else None
        scipy_dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=5,  # Show only the last p merged clusters
            leaf_font_size=10.,
            labels=labels,
        )
        plt.tight_layout()

        # Save the figure to the output directory
        # 将图形保存到输出目录
        output_file = os.path.join(output_dir, f'dendrogram_{linkage}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved as '{output_file}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting dendrogram: {str(e)}")

def plot_dimensionality_reduction(distance_matrix, labels, method='pca', figsize=(10, 8), output_dir='output', linkage='average'):
    """
    Plot dimensionality reduction of the distance matrix
    绘制距离矩阵的降维

    Args:
        distance_matrix (numpy.ndarray): DTW distance matrix
                                        DTW距离矩阵
        labels (numpy.ndarray): Cluster labels
                               簇标签
        method (str, optional): Dimensionality reduction method ('pca' or 'tsne')
                               降维方法（'pca'或'tsne'）
        figsize (tuple, optional): Figure size
                                  图形大小
        output_dir (str, optional): Output directory
                                   输出目录
        linkage (str, optional): Linkage method used
                                使用的连接方法
    """
    try:
        # Apply dimensionality reduction
        # 应用降维
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(distance_matrix)
            title = f'PCA of DTW Distances ({linkage} linkage)'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, metric='precomputed', random_state=42)
            embedding = reducer.fit_transform(distance_matrix)
            title = f't-SNE of DTW Distances ({linkage} linkage)'
        else:
            print(f"Unknown method: {method}. Using PCA.")
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(distance_matrix)
            title = f'PCA of DTW Distances ({linkage} linkage)'

        # Plot the result
        # 绘制结果
        plt.figure(figsize=figsize)

        # Get unique labels and assign colors
        # 获取唯一标签并分配颜色
        unique_labels = np.unique(labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # Plot each cluster with a different color
        # 用不同的颜色绘制每个簇
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Black for noise points
                # 噪声点用黑色
                color = 'k'
                marker = 'x'
            else:
                color = colors[i]
                marker = 'o'

            mask = labels == label
            plt.scatter(
                embedding[mask, 0], embedding[mask, 1],
                color=color, marker=marker,
                label=f'Cluster {label}' if label != -1 else 'Noise',
                alpha=0.8
            )

        plt.title(title, fontsize=15)
        plt.legend()
        plt.tight_layout()

        # Save the figure to the output directory
        # 将图形保存到输出目录
        output_file = os.path.join(output_dir, f'{method.lower()}_visualization_{linkage}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{method.upper()} visualization saved as '{output_file}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting dimensionality reduction: {str(e)}")

def plot_cluster_signals(time_points, signals, labels, figsize=(16, 10), output_dir='output', linkage='average'):
    """
    Plot signals grouped by cluster
    按簇分组绘制信号

    Args:
        time_points (numpy.ndarray): Time points
                                    时间点
        signals (list): List of signal arrays
                       信号数组列表
        labels (numpy.ndarray): Cluster labels
                               簇标签
        figsize (tuple, optional): Figure size
                                  图形大小
        output_dir (str, optional): Output directory
                                   输出目录
        linkage (str, optional): Linkage method used
                                使用的连接方法
    """
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Skip plotting if there's only one cluster or too many clusters
        # 如果只有一个簇或太多簇，则跳过绘图
        if n_clusters == 1:
            print("Only one cluster found. Skipping cluster signals plot.")
            return

        # Determine subplot grid
        # 确定子图网格
        n_cols = min(3, n_clusters)
        n_rows = int(np.ceil(n_clusters / n_cols))

        # Create cluster_plots subdirectory in the output directory
        # 在输出目录中创建cluster_plots子目录
        cluster_plots_dir = os.path.join(output_dir, 'cluster_plots')
        os.makedirs(cluster_plots_dir, exist_ok=True)

        # Plot all clusters
        # 绘制所有簇
        plt.figure(figsize=figsize)

        for i, label in enumerate(unique_labels):
            if label == -1:
                # Skip noise points for visualization clarity
                # 为了可视化清晰，跳过噪声点
                continue

            plt.subplot(n_rows, n_cols, i + 1)

            # Get indices of signals in this cluster
            # 获取该簇中信号的索引
            cluster_indices = np.where(labels == label)[0]

            # Plot each signal in this cluster
            # 绘制该簇中的每个信号
            for idx in cluster_indices:
                plt.plot(time_points, signals[idx], alpha=0.3, linewidth=1)

            # Calculate and plot the mean signal of this cluster
            # 计算并绘制该簇的平均信号
            mean_signal = np.mean([signals[idx] for idx in cluster_indices], axis=0)
            plt.plot(time_points, mean_signal, 'k-', linewidth=2, label='Mean')

            plt.title(f'Cluster {label} (n={len(cluster_indices)})')
            if i == 0:
                plt.legend()

        plt.tight_layout()
        output_file = os.path.join(cluster_plots_dir, f'all_clusters_{linkage}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"All clusters plot saved as '{output_file}'")

        # Plot each cluster individually with more detail
        # 更详细地单独绘制每个簇
        for label in unique_labels:
            if label == -1:
                # Skip noise points for visualization clarity
                # 为了可视化清晰，跳过噪声点
                continue

            plt.figure(figsize=(12, 6))

            # Get indices of signals in this cluster
            # 获取该簇中信号的索引
            cluster_indices = np.where(labels == label)[0]

            # Plot each signal in this cluster
            # 绘制该簇中的每个信号
            for idx in cluster_indices:
                plt.plot(time_points, signals[idx], alpha=0.3, linewidth=1)

            # Calculate and plot the mean signal of this cluster
            # 计算并绘制该簇的平均信号
            mean_signal = np.mean([signals[idx] for idx in cluster_indices], axis=0)
            plt.plot(time_points, mean_signal, 'r-', linewidth=2, label='Mean')

            plt.title(f'Cluster {label} (n={len(cluster_indices)})')
            plt.xlabel('Time')
            plt.ylabel('Signal Value (Z-normalized)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(cluster_plots_dir, f'cluster_{label}_{linkage}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Cluster {label} plot saved as '{output_file}'")
            plt.close()

        plt.close()

        # Plot cluster means for comparison
        # 绘制簇均值进行比较
        plt.figure(figsize=(12, 6))

        for label in unique_labels:
            if label == -1:
                continue

            # Get indices of signals in this cluster
            # 获取该簇中信号的索引
            cluster_indices = np.where(labels == label)[0]

            # Calculate mean signal of this cluster
            # 计算该簇的平均信号
            mean_signal = np.mean([signals[idx] for idx in cluster_indices], axis=0)

            # Plot the mean signal
            # 绘制平均信号
            plt.plot(time_points, mean_signal, linewidth=2, label=f'Cluster {label}')

        plt.title(f'Comparison of Cluster Mean Signals ({linkage} linkage)')
        plt.xlabel('Time')
        plt.ylabel('Signal Value (Z-normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(cluster_plots_dir, f'cluster_means_comparison_{linkage}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cluster means comparison saved as '{output_file}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting cluster signals: {str(e)}")

def plot_evaluation_metrics(evaluation_results, figsize=(12, 8), output_dir='output', linkage='average'):
    """
    Plot evaluation metrics for different numbers of clusters
    绘制不同簇数的评估指标

    Args:
        evaluation_results (dict): Dictionary with evaluation results
                                  包含评估结果的字典
        figsize (tuple, optional): Figure size
                                  图形大小
        output_dir (str, optional): Output directory
                                   输出目录
        linkage (str, optional): Linkage method used
                                使用的连接方法
    """
    try:
        plt.figure(figsize=figsize)

        # Create subplots
        # 创建子图
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Plot silhouette score (higher is better)
        # 绘制轮廓系数（越高越好）
        axs[0].plot(evaluation_results['n_clusters'], evaluation_results['silhouette'],
                    'bo-', linewidth=2)
        axs[0].set_ylabel('Silhouette Score\n(higher is better)')
        axs[0].set_title(f'Clustering Evaluation Metrics ({linkage} linkage)')
        axs[0].grid(True, alpha=0.3)

        # Highlight the best value
        # 突出显示最佳值
        best_idx = np.argmax(evaluation_results['silhouette'])
        best_n = evaluation_results['n_clusters'][best_idx]
        best_score = evaluation_results['silhouette'][best_idx]
        axs[0].scatter([best_n], [best_score], c='r', s=100, marker='*',
                       label=f'Best: {best_n} clusters')
        axs[0].legend()

        # Plot Davies-Bouldin index (lower is better)
        # 绘制Davies-Bouldin指数（越低越好）
        axs[1].plot(evaluation_results['n_clusters'], evaluation_results['davies_bouldin'],
                    'go-', linewidth=2)
        axs[1].set_ylabel('Davies-Bouldin Index\n(lower is better)')
        axs[1].grid(True, alpha=0.3)

        # Highlight the best value
        # 突出显示最佳值
        best_idx = np.argmin(evaluation_results['davies_bouldin'])
        best_n = evaluation_results['n_clusters'][best_idx]
        best_score = evaluation_results['davies_bouldin'][best_idx]
        axs[1].scatter([best_n], [best_score], c='r', s=100, marker='*',
                       label=f'Best: {best_n} clusters')
        axs[1].legend()

        # Plot Calinski-Harabasz index (higher is better)
        # 绘制Calinski-Harabasz指数（越高越好）
        axs[2].plot(evaluation_results['n_clusters'], evaluation_results['calinski_harabasz'],
                    'mo-', linewidth=2)
        axs[2].set_xlabel('Number of Clusters')
        axs[2].set_ylabel('Calinski-Harabasz Index\n(higher is better)')
        axs[2].grid(True, alpha=0.3)

        # Highlight the best value
        # 突出显示最佳值
        best_idx = np.argmax(evaluation_results['calinski_harabasz'])
        best_n = evaluation_results['n_clusters'][best_idx]
        best_score = evaluation_results['calinski_harabasz'][best_idx]
        axs[2].scatter([best_n], [best_score], c='r', s=100, marker='*',
                       label=f'Best: {best_n} clusters')
        axs[2].legend()

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'evaluation_metrics_{linkage}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Evaluation metrics plot saved as '{output_file}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting evaluation metrics: {str(e)}")

def plot_linkage_comparison(all_results, figsize=(15, 12), output_dir='output'):
    """
    Plot comparison of evaluation metrics for different linkage methods
    绘制不同连接方法的评估指标比较

    Args:
        all_results (dict): Dictionary with results for each linkage method
                           包含每种连接方法结果的字典
        figsize (tuple, optional): Figure size
                                  图形大小
        output_dir (str, optional): Output directory
                                   输出目录
    """
    try:
        # Create subplots
        # 创建子图
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Get unique cluster numbers across all methods
        # 获取所有方法中的唯一簇数
        all_n_clusters = set()
        for results in all_results.values():
            all_n_clusters.update(results['n_clusters'])
        all_n_clusters = sorted(list(all_n_clusters))

        # Plot silhouette score comparison
        # 绘制轮廓系数比较
        for linkage, results in all_results.items():
            # Create interpolated values for consistent x-axis
            # 创建插值，使x轴一致
            axs[0].plot(results['n_clusters'], results['silhouette'],
                        'o-', linewidth=2, label=linkage)

            # Highlight the best value for each method
            # 突出显示每种方法的最佳值
            best_idx = np.argmax(results['silhouette'])
            best_n = results['n_clusters'][best_idx]
            best_score = results['silhouette'][best_idx]
            axs[0].scatter([best_n], [best_score], s=100, marker='*')

        axs[0].set_ylabel('Silhouette Score\n(higher is better)')
        axs[0].set_title('Comparison of Clustering Evaluation Metrics Across Linkage Methods')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(title='Linkage Method')

        # Plot Davies-Bouldin index comparison
        # 绘制Davies-Bouldin指数比较
        for linkage, results in all_results.items():
            axs[1].plot(results['n_clusters'], results['davies_bouldin'],
                        'o-', linewidth=2, label=linkage)

            # Highlight the best value for each method
            # 突出显示每种方法的最佳值
            best_idx = np.argmin(results['davies_bouldin'])
            best_n = results['n_clusters'][best_idx]
            best_score = results['davies_bouldin'][best_idx]
            axs[1].scatter([best_n], [best_score], s=100, marker='*')

        axs[1].set_ylabel('Davies-Bouldin Index\n(lower is better)')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(title='Linkage Method')

        # Plot Calinski-Harabasz index comparison
        # 绘制Calinski-Harabasz指数比较
        for linkage, results in all_results.items():
            axs[2].plot(results['n_clusters'], results['calinski_harabasz'],
                        'o-', linewidth=2, label=linkage)

            # Highlight the best value for each method
            # 突出显示每种方法的最佳值
            best_idx = np.argmax(results['calinski_harabasz'])
            best_n = results['n_clusters'][best_idx]
            best_score = results['calinski_harabasz'][best_idx]
            axs[2].scatter([best_n], [best_score], s=100, marker='*')

        axs[2].set_xlabel('Number of Clusters')
        axs[2].set_ylabel('Calinski-Harabasz Index\n(higher is better)')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(title='Linkage Method')

        plt.tight_layout()
        output_file = os.path.join(output_dir, 'linkage_comparison_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Linkage comparison plot saved as '{output_file}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting linkage comparison: {str(e)}")