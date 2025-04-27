import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import modules
from data_loader import load_data, preprocess_signals
from dtw_calculator import calculate_dtw_distance_matrix
from clustering import hierarchical_clustering, time_series_kmeans, dbscan_clustering
from cluster_evaluation import evaluate_clustering, find_optimal_clusters
from visualization import (
    plot_dendrogram,
    plot_dimensionality_reduction,
    plot_cluster_signals,
    plot_evaluation_metrics
)
from utils import (
    print_clustering_summary,
    save_results,
    ensure_output_directory,
    plot_signals_overview
)


def main():
    """
    Main function to run the time series clustering pipeline
    运行时间序列聚类流程的主函数
    """
    # Parse command line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Time Series Clustering using DTW')
    parser.add_argument('--file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--max_clusters', type=int, default=10, help='Maximum number of clusters to try')
    parser.add_argument('--sakoe_chiba_radius', type=int, default=None,
                         help='Sakoe-Chiba band radius (None = no constraint)')
    parser.add_argument('--linkage', type=str, default='ward',
                         choices=['ward', 'complete', 'average', 'single'],
                         help='Linkage criterion for hierarchical clustering')
    parser.add_argument('--handle_missing', type=str, default='report',
                         choices=['report', 'drop', 'fill_mean', 'fill_zero'],
                         help='Strategy for handling missing values')
    parser.add_argument('--linkage', type=str, default='average',
                        choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage criterion for hierarchical clustering. Note: when using DTW distances, "ward" will be automatically changed to "average"')
    args = parser.parse_args()

    # Ensure output directories exist
    # 确保输出目录存在
    ensure_output_directory()

    # Step 1: Load and preprocess data
    # 步骤1：加载和预处理数据
    print("\n===== Step 1: Loading and preprocessing data =====")
    time_points, signal_names, signals = load_data(args.file, handle_missing=args.handle_missing)
    print(f"Loaded {len(signals)} signals with {len(time_points)} time points each")

    # Plot signals overview
    # 绘制信号概览
    plot_signals_overview(time_points, signals, signal_names, n_signals=10)

    # Preprocess signals (Z-score normalization)
    # 预处理信号（Z分数标准化）
    preprocessed_signals = preprocess_signals(signals)
    print("Signals preprocessed with Z-score normalization")

    # Step 2: Calculate DTW distance matrix
    # 步骤2：计算DTW距离矩阵
    print("\n===== Step 2: Calculating DTW distance matrix =====")
    dtw_matrix = calculate_dtw_distance_matrix(preprocessed_signals, args.sakoe_chiba_radius)

    # Step 3: Find optimal number of clusters
    # 步骤3：寻找最优簇数
    print("\n===== Step 3: Finding optimal number of clusters =====")
    evaluation_results, best_n_clusters = find_optimal_clusters(
        dtw_matrix,
        preprocessed_signals,
        max_clusters=args.max_clusters
    )

    # Plot evaluation metrics
    # 绘制评估指标
    plot_evaluation_metrics(evaluation_results)

    # Step 4: Perform hierarchical clustering with optimal number of clusters
    # 步骤4：使用最优簇数执行层次聚类
    print(f"\n===== Step 4: Performing hierarchical clustering with {best_n_clusters} clusters =====")
    hier_model = hierarchical_clustering(dtw_matrix, n_clusters=best_n_clusters, linkage=args.linkage)
    hier_labels = hier_model.labels_

    # Plot dendrogram (unclustered, for all hierarchical structure)
    # 绘制树状图（未聚类，显示所有层次结构）
    unclustered_model = hierarchical_clustering(dtw_matrix, n_clusters=None, linkage=args.linkage)
    plot_dendrogram(unclustered_model, distance_matrix=dtw_matrix, signal_names=signal_names)

    # Step 5: Visualize clustering results
    # 步骤5：可视化聚类结果
    print("\n===== Step 5: Visualizing clustering results =====")

    # Plot dimensionality reduction
    # 绘制降维结果
    print("Generating PCA visualization...")
    plot_dimensionality_reduction(dtw_matrix, hier_labels, method='pca')

    print("Generating t-SNE visualization...")
    plot_dimensionality_reduction(dtw_matrix, hier_labels, method='tsne')

    # Plot cluster signals
    # 绘制簇信号
    print("Generating cluster signal plots...")
    plot_cluster_signals(time_points, preprocessed_signals, hier_labels)

    # Step 6: Try TimeSeriesKMeans as alternative method
    # 步骤6：尝试TimeSeriesKMeans作为替代方法
    print(f"\n===== Step 6: Trying TimeSeriesKMeans with {best_n_clusters} clusters =====")
    tskm_model = time_series_kmeans(preprocessed_signals, n_clusters=best_n_clusters)
    tskm_labels = tskm_model.labels_

    # Calculate silhouette score for TimeSeriesKMeans
    # 计算TimeSeriesKMeans的轮廓系数
    tskm_silhouette = evaluate_clustering(dtw_matrix, tskm_labels)
    hier_silhouette = evaluate_clustering(dtw_matrix, hier_labels)

    print(f"Hierarchical Clustering Silhouette Score: {hier_silhouette:.4f}")
    print(f"TimeSeriesKMeans Silhouette Score: {tskm_silhouette:.4f}")

    # Decide which clustering result to use based on silhouette score
    # 基于轮廓系数决定使用哪个聚类结果
    if tskm_silhouette > hier_silhouette:
        print("TimeSeriesKMeans produces better clusters. Using these results.")
        final_labels = tskm_labels
        final_method = "TimeSeriesKMeans"
    else:
        print("Hierarchical clustering produces better clusters. Using these results.")
        final_labels = hier_labels
        final_method = "Hierarchical Clustering"

    # Step 7: Print and save final results
    # 步骤7：打印和保存最终结果
    print(f"\n===== Step 7: Final clustering results ({final_method}) =====")
    print_clustering_summary(final_labels, signal_names)
    save_results(final_labels, signal_names)

    print("\n===== Clustering completed successfully =====")
    print(f"Recommended number of clusters: {best_n_clusters}")
    print(f"Best clustering method: {final_method}")
    print("\nResults and visualizations have been saved to disk.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)