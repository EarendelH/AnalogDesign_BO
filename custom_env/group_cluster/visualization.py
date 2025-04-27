import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os
import sys


def plot_dendrogram(model, signal_names=None, figsize=(12, 8)):
    """
    Plot dendrogram from hierarchical clustering
    绘制层次聚类的树状图

    Args:
        model: Fitted AgglomerativeClustering model
               拟合的AgglomerativeClustering模型
        signal_names (list, optional): List of signal names
                                      信号名称列表
        figsize (tuple, optional): Figure size
                                  图形大小
    """
    try:
        # Create linkage matrix from the children attribute
        # 从children属性创建连接矩阵
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([
            model.children_, model.distances_, counts
        ]).astype(float)

        # Plot the dendrogram
        # 绘制树状图
        plt.figure(figsize=figsize)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=15)
        plt.xlabel('Signal Index or Signal Name', fontsize=12)
        plt.ylabel('Distance', fontsize=12)

        labels = signal_names if signal_names else None
        dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=5,  # Show only the last p merged clusters
            leaf_font_size=10.,
            labels=labels,
        )
        plt.tight_layout()

        # Save the figure
        # 保存图形
        plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
        print("Dendrogram saved as 'dendrogram.png'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting dendrogram: {str(e)}")


def plot_dimensionality_reduction(distance_matrix, labels, method='pca', figsize=(10, 8)):
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
    """
    try:
        # Convert distances to features for dimensionality reduction
        # 将距离转换为降维的特征

        # Apply dimensionality reduction
        # 应用降维
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(distance_matrix)
            title = 'PCA of DTW Distances'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, metric='precomputed', random_state=42)
            embedding = reducer.fit_transform(distance_matrix)
            title = 't-SNE of DTW Distances'
        else:
            print(f"Unknown method: {method}. Using PCA.")
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(distance_matrix)
            title = 'PCA of DTW Distances'

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

        # Save the figure
        # 保存图形
        filename = f"{method.lower()}_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"{method.upper()} visualization saved as '{filename}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting dimensionality reduction: {str(e)}")


def plot_cluster_signals(time_points, signals, labels, figsize=(16, 10)):
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

        # Create a directory for cluster plots
        # 创建存放簇图的目录
        os.makedirs('cluster_plots', exist_ok=True)

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
        plt.savefig('cluster_plots/all_clusters.png', dpi=300, bbox_inches='tight')
        print("All clusters plot saved as 'cluster_plots/all_clusters.png'")

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
            plt.savefig(f'cluster_plots/cluster_{label}.png', dpi=300, bbox_inches='tight')
            print(f"Cluster {label} plot saved as 'cluster_plots/cluster_{label}.png'")
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

        plt.title('Comparison of Cluster Mean Signals')
        plt.xlabel('Time')
        plt.ylabel('Signal Value (Z-normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('cluster_plots/cluster_means_comparison.png', dpi=300, bbox_inches='tight')
        print("Cluster means comparison saved as 'cluster_plots/cluster_means_comparison.png'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting cluster signals: {str(e)}")


def plot_evaluation_metrics(evaluation_results, figsize=(12, 8)):
    """
    Plot evaluation metrics for different numbers of clusters
    绘制不同簇数的评估指标

    Args:
        evaluation_results (dict): Dictionary with evaluation results
                                  包含评估结果的字典
        figsize (tuple, optional): Figure size
                                  图形大小
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
        axs[0].set_title('Clustering Evaluation Metrics')
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
        plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
        print("Evaluation metrics plot saved as 'evaluation_metrics.png'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting evaluation metrics: {str(e)}")