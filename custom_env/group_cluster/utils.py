import numpy as np
import os
import sys
import matplotlib.pyplot as plt


def print_clustering_summary(labels, signal_names=None):
    """
    Print a summary of clustering results
    打印聚类结果的摘要

    Args:
        labels (numpy.ndarray): Cluster labels
                               簇标签
        signal_names (list, optional): List of signal names
                                      信号名称列表
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    print("\n===== Clustering Summary =====")
    print(f"Number of clusters: {n_clusters}")

    # Print cluster sizes
    # 打印簇大小
    print("\nCluster sizes:")
    for label in sorted(unique_labels):
        if label == -1:
            n_members = np.sum(labels == label)
            print(f"  Noise points: {n_members}")
        else:
            n_members = np.sum(labels == label)
            print(f"  Cluster {label}: {n_members} signals")

    # Print cluster members if signal names are provided
    # 如果提供了信号名称，则打印簇成员
    if signal_names:
        print("\nCluster members:")
        for label in sorted(unique_labels):
            if label == -1:
                print("  Noise points:")
                noise_indices = np.where(labels == label)[0]
                for idx in noise_indices:
                    print(f"    - {signal_names[idx]}")
            else:
                print(f"  Cluster {label}:")
                cluster_indices = np.where(labels == label)[0]
                for idx in cluster_indices:
                    print(f"    - {signal_names[idx]}")


def save_results(labels, signal_names=None, output_dir='output', linkage='average'):
    """
    Save clustering results to a text file
    将聚类结果保存到文本文件

    Args:
        labels (numpy.ndarray): Cluster labels
                               簇标签
        signal_names (list, optional): List of signal names
                                      信号名称列表
        output_dir (str, optional): Output directory
                                   输出目录
        linkage (str, optional): Linkage method used
                                使用的连接方法
    """
    try:
        # Create output file path
        # 创建输出文件路径
        output_file = os.path.join(output_dir, f'clustering_results_{linkage}.txt')

        with open(output_file, 'w') as f:
            f.write(f"===== Clustering Results (Linkage: {linkage}) =====\n\n")

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            f.write(f"Number of clusters: {n_clusters}\n\n")

            # Write cluster sizes
            # 写入簇大小
            f.write("Cluster sizes:\n")
            for label in sorted(unique_labels):
                if label == -1:
                    n_members = np.sum(labels == label)
                    f.write(f"  Noise points: {n_members}\n")
                else:
                    n_members = np.sum(labels == label)
                    f.write(f"  Cluster {label}: {n_members} signals\n")

            # Write cluster members if signal names are provided
            # 如果提供了信号名称，则写入簇成员
            if signal_names:
                f.write("\nCluster members:\n")
                for label in sorted(unique_labels):
                    if label == -1:
                        f.write("  Noise points:\n")
                        noise_indices = np.where(labels == label)[0]
                        for idx in noise_indices:
                            f.write(f"    - {signal_names[idx]}\n")
                    else:
                        f.write(f"  Cluster {label}:\n")
                        cluster_indices = np.where(labels == label)[0]
                        for idx in cluster_indices:
                            f.write(f"    - {signal_names[idx]}\n")

        print(f"Clustering results saved to '{output_file}'")

    except Exception as e:
        print(f"Error in saving results: {str(e)}")


def ensure_output_directory(max_clusters, linkage='average'):
    """
    Ensure output directories exist with control options in the name
    确保带有控制选项的输出目录存在

    Args:
        max_clusters (int): Maximum number of clusters
                           最大簇数
        linkage (str, optional): Linkage method ('all' for all methods)
                                连接方法（'all'表示所有方法）

    Returns:
        str: Path to the output directory
             输出目录路径
    """
    # Create output directory name with control options
    # 创建带有控制选项的输出目录名
    dir_name = f"output_maxk{max_clusters}_linkage{linkage}"

    # Create output directories
    # 创建输出目录
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, 'cluster_plots'), exist_ok=True)

    print(f"Output directories created: {dir_name}")
    return dir_name


def plot_signals_overview(time_points, signals, signal_names=None, n_signals=None, figsize=(14, 8),
                          output_dir='output'):
    """
    Plot an overview of signals
    绘制信号概览

    Args:
        time_points (numpy.ndarray): Time points
                                    时间点
        signals (list): List of signal arrays
                       信号数组列表
        signal_names (list, optional): List of signal names
                                      信号名称列表
        n_signals (int, optional): Number of signals to plot (None = all)
                                  绘制的信号数量（None = 全部）
        figsize (tuple, optional): Figure size
                                  图形大小
        output_dir (str, optional): Output directory
                                   输出目录
    """
    try:
        # Determine how many signals to plot
        # 确定要绘制的信号数量
        if n_signals is None or n_signals > len(signals):
            n_signals = len(signals)

        plt.figure(figsize=figsize)

        # Plot each signal
        # 绘制每个信号
        for i in range(n_signals):
            if signal_names:
                plt.plot(time_points, signals[i], label=signal_names[i])
            else:
                plt.plot(time_points, signals[i], label=f'Signal {i}')

        plt.title('Overview of Time Series Signals (Z-normalized)')
        plt.xlabel('Time')
        plt.ylabel('Signal Value (Z-normalized)')

        # Add legend if not too many signals
        # 如果信号不太多，则添加图例
        if n_signals <= 10:
            plt.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the figure to the output directory
        # 将图形保存到输出目录
        output_file = os.path.join(output_dir, 'signals_overview.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Signals overview saved as '{output_file}'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting signals overview: {str(e)}")


def create_linkage_comparison_table(all_results, output_dir='output'):
    """
    Create a comparison table for different linkage methods
    创建不同连接方法的比较表

    Args:
        all_results (dict): Dictionary with results for each linkage method
                           包含每种连接方法结果的字典
        output_dir (str, optional): Output directory
                                   输出目录

    Returns:
        tuple: (best_method, best_n_clusters) - Best linkage method and number of clusters
               (最佳连接方法，最佳簇数量)
    """
    try:
        # Get the best number of clusters for each method based on silhouette score
        # 基于轮廓系数获取每种方法的最佳簇数
        best_n_clusters = {}
        best_scores = {
            'silhouette': {},
            'davies_bouldin': {},
            'calinski_harabasz': {}
        }

        for linkage, results in all_results.items():
            # Get the index of the maximum silhouette score
            # 获取最大轮廓系数的索引
            best_idx = np.argmax(results['silhouette'])
            best_n_clusters[linkage] = results['n_clusters'][best_idx]

            # Get the best scores for each metric
            # 获取每个指标的最佳分数
            best_scores['silhouette'][linkage] = max(results['silhouette'])
            best_scores['davies_bouldin'][linkage] = min(results['davies_bouldin'])
            best_scores['calinski_harabasz'][linkage] = max(results['calinski_harabasz'])

        # Create comparison table
        # 创建比较表
        output_file = os.path.join(output_dir, 'linkage_comparison.txt')
        with open(output_file, 'w') as f:
            f.write("===== Linkage Methods Comparison =====\n\n")

            # Table header
            # 表格标题
            f.write(
                "Linkage Method | Best Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index\n")
            f.write(
                "---------------|---------------|------------------|---------------------|------------------------\n")

            # Table rows
            # 表格行
            for linkage in all_results.keys():
                f.write(f"{linkage.ljust(15)}| {str(best_n_clusters[linkage]).ljust(15)}| "
                        f"{best_scores['silhouette'][linkage]:.4f}".ljust(18) + "| "
                                                                                f"{best_scores['davies_bouldin'][linkage]:.4f}".ljust(
                    21) + "| "
                          f"{best_scores['calinski_harabasz'][linkage]:.4f}\n")

            # Identify the best method for each metric
            # 确定每个指标的最佳方法
            best_silhouette_method = max(best_scores['silhouette'], key=best_scores['silhouette'].get)
            best_db_method = min(best_scores['davies_bouldin'], key=best_scores['davies_bouldin'].get)
            best_ch_method = max(best_scores['calinski_harabasz'], key=best_scores['calinski_harabasz'].get)

            f.write("\nBest methods for each evaluation metric:\n")
            f.write(
                f"Silhouette Score (higher is better): {best_silhouette_method} ({best_scores['silhouette'][best_silhouette_method]:.4f})\n")
            f.write(
                f"Davies-Bouldin Index (lower is better): {best_db_method} ({best_scores['davies_bouldin'][best_db_method]:.4f})\n")
            f.write(
                f"Calinski-Harabasz Index (higher is better): {best_ch_method} ({best_scores['calinski_harabasz'][best_ch_method]:.4f})\n")

        print(f"Linkage comparison table saved to '{output_file}'")

        return best_silhouette_method, best_n_clusters[best_silhouette_method]

    except Exception as e:
        print(f"Error in creating linkage comparison table: {str(e)}")
        return None, None