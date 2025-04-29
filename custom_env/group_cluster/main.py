import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import modules
from data_loader import load_data, preprocess_signals
from dtw_calculator import calculate_dtw_distance_matrix
from clustering import hierarchical_clustering
from cluster_evaluation import evaluate_clustering, find_optimal_clusters, evaluate_all_linkage_methods
from visualization import (
    plot_dendrogram,
    plot_dimensionality_reduction,
    plot_cluster_signals,
    plot_evaluation_metrics,
    plot_linkage_comparison
)
from utils import (
    print_clustering_summary,
    save_results,
    ensure_output_directory,
    plot_signals_overview,
    create_linkage_comparison_table
)


def main():
    """
    Main function to run the time series clustering pipeline
    运行时间序列聚类流程的主函数
    """
    # Parse command line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Time Series Clustering using DTW')

    # Data input parameters
    # 数据输入参数
    parser.add_argument('--file', type=str, required=True, help='Path to CSV file')

    # DTW parameters
    # DTW参数
    parser.add_argument('--sakoe_chiba_radius', type=int, default=None,
                        help='Sakoe-Chiba band radius (None = no constraint)')

    # Hierarchical clustering parameters
    # 层次聚类参数
    parser.add_argument('--linkage', type=str, default='average',
                        choices=['ward', 'complete', 'average', 'single', 'all'],
                        help='Linkage criterion for hierarchical clustering (all = try all methods)')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum number of clusters to try')

    args = parser.parse_args()

    # Create output directory with control options in the name
    # 创建包含控制选项的输出目录
    output_dir = ensure_output_directory(args.max_clusters, args.linkage)

    # Step 1: Load and preprocess data
    # 步骤1：加载和预处理数据
    print("\n===== Step 1: Loading and preprocessing data =====")
    time_points, signal_names, signals = load_data(args.file)
    print(f"Loaded {len(signals)} signals with {len(time_points)} time points each")

    # Plot signals overview
    # 绘制信号概览
    plot_signals_overview(time_points, signals, signal_names, n_signals=10, output_dir=output_dir)

    # Preprocess signals (Z-score normalization)
    # 预处理信号（Z分数标准化）
    preprocessed_signals = preprocess_signals(signals)
    print("Signals preprocessed with Z-score normalization")

    # Step 2: Calculate DTW distance matrix
    # 步骤2：计算DTW距离矩阵
    print("\n===== Step 2: Calculating DTW distance matrix =====")
    dtw_matrix = calculate_dtw_distance_matrix(preprocessed_signals, args.sakoe_chiba_radius)

    # Dictionary to store clustering results
    # 存储聚类结果的字典
    clustering_results = {}

    # Step 3: Determine optimal number of clusters for each linkage method
    # 步骤3：确定每种连接方法的最优簇数
    print("\n===== Step 3: Finding optimal number of clusters =====")

    if args.linkage == 'all':
        # Evaluate all linkage methods
        # 评估所有连接方法
        all_results = evaluate_all_linkage_methods(dtw_matrix, preprocessed_signals, max_clusters=args.max_clusters)

        # Plot comparison of linkage methods
        # 绘制连接方法的比较
        plot_linkage_comparison(all_results, output_dir=output_dir)

        # Create comparison table and determine overall best method
        # 创建比较表并确定总体最佳方法
        best_linkage, best_n_clusters = create_linkage_comparison_table(all_results, output_dir=output_dir)

        print(f"\nBest linkage method: {best_linkage}")
        print(f"Best number of clusters: {best_n_clusters}")

        # Perform clustering for each linkage method with its optimal number of clusters
        # 使用每种连接方法的最优簇数执行聚类
        for linkage, results in all_results.items():
            # Get the best number of clusters for this linkage method
            # 获取该连接方法的最佳簇数
            best_idx = np.argmax(results['silhouette'])
            n_clusters = results['n_clusters'][best_idx]

            print(f"\n===== Performing hierarchical clustering with {linkage} linkage and {n_clusters} clusters =====")
            model = hierarchical_clustering(dtw_matrix, n_clusters=n_clusters, linkage=linkage)
            labels = model.labels_
            silhouette = evaluate_clustering(dtw_matrix, labels)
            print(f"Hierarchical Clustering ({linkage}) Silhouette Score: {silhouette:.4f}")

            # Store results for this linkage method
            # 存储该连接方法的结果
            clustering_results[linkage] = {
                'model': model,
                'labels': labels,
                'silhouette': silhouette,
                'n_clusters': n_clusters
            }

            # Plot evaluation metrics for this linkage method
            # 绘制该连接方法的评估指标
            plot_evaluation_metrics(results, output_dir=output_dir, linkage=linkage)

            # Plot dendrogram (unclustered, for all hierarchical structure)
            # 绘制树状图（未聚类，显示所有层次结构）
            print(f"Creating dendrogram for {linkage} linkage...")
            unclustered_model = hierarchical_clustering(
                dtw_matrix,
                n_clusters=None,
                linkage=linkage,
                distance_threshold=0.0
            )
            plot_dendrogram(unclustered_model, distance_matrix=dtw_matrix, signal_names=signal_names,
                            output_dir=output_dir, linkage=linkage)

            # Visualize clustering results for this linkage method
            # 可视化该连接方法的聚类结果
            print(f"Generating visualizations for {linkage} linkage...")
            plot_dimensionality_reduction(dtw_matrix, labels, method='pca',
                                          output_dir=output_dir, linkage=linkage)
            plot_dimensionality_reduction(dtw_matrix, labels, method='tsne',
                                          output_dir=output_dir, linkage=linkage)
            plot_cluster_signals(time_points, preprocessed_signals, labels,
                                 output_dir=output_dir, linkage=linkage)

            # Print and save results for this linkage method
            # 打印和保存该连接方法的结果
            print(f"\n===== Clustering results for {linkage} linkage =====")
            print_clustering_summary(labels, signal_names)
            save_results(labels, signal_names, output_dir=output_dir, linkage=linkage)

        # Step 4: Use the overall best method for the final results
        # 步骤4：使用总体最佳方法作为最终结果
        final_labels = clustering_results[best_linkage]['labels']
        final_method = f"Hierarchical Clustering with {best_linkage} linkage"
        final_silhouette = clustering_results[best_linkage]['silhouette']

    else:
        # Single linkage method specified
        # 指定了单一连接方法
        evaluation_results, best_n_clusters = find_optimal_clusters(
            dtw_matrix,
            preprocessed_signals,
            max_clusters=args.max_clusters,
            linkage=args.linkage
        )

        # Plot evaluation metrics
        # 绘制评估指标
        plot_evaluation_metrics(evaluation_results, output_dir=output_dir, linkage=args.linkage)
        print(f"Recommended number of clusters: {best_n_clusters}")

        # Step 4: Perform hierarchical clustering with the specified linkage method
        # 步骤4：使用指定的连接方法执行层次聚类
        print(
            f"\n===== Performing hierarchical clustering with {args.linkage} linkage and {best_n_clusters} clusters =====")
        model = hierarchical_clustering(dtw_matrix, n_clusters=best_n_clusters, linkage=args.linkage)
        labels = model.labels_
        silhouette = evaluate_clustering(dtw_matrix, labels)
        print(f"Hierarchical Clustering Silhouette Score: {silhouette:.4f}")

        # Plot dendrogram (unclustered, for all hierarchical structure)
        # 绘制树状图（未聚类，显示所有层次结构）
        print("Creating model for dendrogram...")
        unclustered_model = hierarchical_clustering(
            dtw_matrix,
            n_clusters=None,
            linkage=args.linkage,
            distance_threshold=0.0
        )
        plot_dendrogram(unclustered_model, distance_matrix=dtw_matrix, signal_names=signal_names,
                        output_dir=output_dir, linkage=args.linkage)

        final_labels = labels
        final_method = f"Hierarchical Clustering with {args.linkage} linkage"
        final_silhouette = silhouette

    # Step 5: Visualize final clustering results
    # 步骤5：可视化最终聚类结果
    print("\n===== Visualizing final clustering results =====")

    if args.linkage != 'all':
        # Visualizations already created for 'all' linkage methods
        # 'all'连接方法的可视化已经创建
        print("Generating PCA visualization...")
        plot_dimensionality_reduction(dtw_matrix, final_labels, method='pca',
                                      output_dir=output_dir, linkage=args.linkage)

        print("Generating t-SNE visualization...")
        plot_dimensionality_reduction(dtw_matrix, final_labels, method='tsne',
                                      output_dir=output_dir, linkage=args.linkage)

        # Plot cluster signals
        # 绘制簇信号
        print("Generating cluster signal plots...")
        plot_cluster_signals(time_points, preprocessed_signals, final_labels,
                             output_dir=output_dir, linkage=args.linkage)

    # Step 6: Print final results
    # 步骤6：打印最终结果
    print(f"\n===== Final clustering results ({final_method}) =====")
    if args.linkage != 'all':
        # Results already printed and saved for 'all' linkage methods
        # 'all'连接方法的结果已经打印和保存
        print_clustering_summary(final_labels, signal_names)
        save_results(final_labels, signal_names, output_dir=output_dir, linkage=args.linkage)

    print("\n===== Clustering completed successfully =====")
    print(f"Best clustering method: {final_method} (Silhouette Score: {final_silhouette:.4f})")
    print(f"\nResults and visualizations have been saved to the directory: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)