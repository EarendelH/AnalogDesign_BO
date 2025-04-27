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


def save_results(labels, signal_names=None, output_file='clustering_results.txt'):
    """
    Save clustering results to a text file
    将聚类结果保存到文本文件

    Args:
        labels (numpy.ndarray): Cluster labels
                               簇标签
        signal_names (list, optional): List of signal names
                                      信号名称列表
        output_file (str, optional): Output file path
                                    输出文件路径
    """
    try:
        with open(output_file, 'w') as f:
            f.write("===== Clustering Results =====\n\n")

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


def ensure_output_directory():
    """
    Ensure output directories exist
    确保输出目录存在
    """
    os.makedirs('cluster_plots', exist_ok=True)
    print("Output directories created")


def plot_signals_overview(time_points, signals, signal_names=None, n_signals=None, figsize=(14, 8)):
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

        plt.savefig('signals_overview.png', dpi=300, bbox_inches='tight')
        print("Signals overview saved as 'signals_overview.png'")

        plt.close()

    except Exception as e:
        print(f"Error in plotting signals overview: {str(e)}")