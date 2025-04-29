import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

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


def process_single_file(file_path, sakoe_chiba_radius, linkage, max_clusters, n_clusters, output_dir):
    """
    Process a single CSV file for time series clustering
    处理单个CSV文件进行时间序列聚类

    Args:
        file_path (str): Path to the CSV file
                        CSV文件路径
        sakoe_chiba_radius (int): Sakoe-Chiba band radius
                                Sakoe-Chiba带半径
        linkage (str): Linkage criterion for hierarchical clustering
                      层次聚类的连接方法
        max_clusters (int): Maximum number of clusters to try
                           尝试的最大簇数
        n_clusters (int): Manually specified number of clusters (or None)
                         手动指定的簇数（或None）
        output_dir (str): Output directory for results
                         结果的输出目录

    Returns:
        tuple: (final_method, final_silhouette) - Final clustering method and silhouette score
               (最终聚类方法, 最终轮廓系数)
    """
    # Step 1: Load and preprocess data
    # 步骤1：加载和预处理数据
    print(f"\n===== Processing file: {file_path} =====")
    print("\n===== Step 1: Loading and preprocessing data =====")
    time_points, signal_names, signals = load_data(file_path)
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
    dtw_matrix = calculate_dtw_distance_matrix(preprocessed_signals, sakoe_chiba_radius)

    # Dictionary to store clustering results
    # 存储聚类结果的字典
    clustering_results = {}

    # Check if user manually specified the number of clusters
    # 检查用户是否手动指定了簇的数量
    if n_clusters is not None:
        print(f"\n===== Using manually specified number of clusters: {n_clusters} =====")
        manual_n_clusters = n_clusters

        # Validate the specified number of clusters
        # 验证指定的簇数量
        if manual_n_clusters < 2:
            print("Error: Number of clusters must be at least 2.")
            sys.exit(1)
        if manual_n_clusters > len(signals):
            print(
                f"Error: Number of clusters ({manual_n_clusters}) cannot exceed the number of signals ({len(signals)}).")
            sys.exit(1)

        # Skip the optimal cluster number detection step
        # 跳过最优簇数检测步骤
        best_n_clusters = manual_n_clusters
    else:
        # Step 3: Determine optimal number of clusters for each linkage method
        # 步骤3：确定每种连接方法的最优簇数
        print("\n===== Step 3: Finding optimal number of clusters =====")

        if linkage == 'all':
            # Evaluate all linkage methods
            # 评估所有连接方法
            all_results = evaluate_all_linkage_methods(dtw_matrix, preprocessed_signals, max_clusters=max_clusters)

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
            for link_method, results in all_results.items():
                # Get the best number of clusters for this linkage method
                # 获取该连接方法的最佳簇数
                best_idx = np.argmax(results['silhouette'])
                n_clusters_opt = results['n_clusters'][best_idx]

                print(
                    f"\n===== Performing hierarchical clustering with {link_method} linkage and {n_clusters_opt} clusters =====")
                model = hierarchical_clustering(dtw_matrix, n_clusters=n_clusters_opt, linkage=link_method)
                labels = model.labels_
                silhouette = evaluate_clustering(dtw_matrix, labels)
                print(f"Hierarchical Clustering ({link_method}) Silhouette Score: {silhouette:.4f}")

                # Store results for this linkage method
                # 存储该连接方法的结果
                clustering_results[link_method] = {
                    'model': model,
                    'labels': labels,
                    'silhouette': silhouette,
                    'n_clusters': n_clusters_opt
                }

                # Plot evaluation metrics for this linkage method
                # 绘制该连接方法的评估指标
                plot_evaluation_metrics(results, output_dir=output_dir, linkage=link_method)

                # Plot dendrogram (unclustered, for all hierarchical structure)
                # 绘制树状图（未聚类，显示所有层次结构）
                print(f"Creating dendrogram for {link_method} linkage...")
                unclustered_model = hierarchical_clustering(
                    dtw_matrix,
                    n_clusters=None,
                    linkage=link_method,
                    distance_threshold=0.0
                )
                plot_dendrogram(unclustered_model, distance_matrix=dtw_matrix, signal_names=signal_names,
                                output_dir=output_dir, linkage=link_method)

                # Visualize clustering results for this linkage method
                # 可视化该连接方法的聚类结果
                print(f"Generating visualizations for {link_method} linkage...")
                plot_dimensionality_reduction(dtw_matrix, labels, method='pca',
                                              output_dir=output_dir, linkage=link_method)
                plot_dimensionality_reduction(dtw_matrix, labels, method='tsne',
                                              output_dir=output_dir, linkage=link_method)
                plot_cluster_signals(time_points, preprocessed_signals, labels,
                                     output_dir=output_dir, linkage=link_method)

                # Print and save results for this linkage method
                # 打印和保存该连接方法的结果
                print(f"\n===== Clustering results for {link_method} linkage =====")
                print_clustering_summary(labels, signal_names)
                save_results(labels, signal_names, output_dir=output_dir, linkage=link_method)

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
                max_clusters=max_clusters,
                linkage=linkage
            )

            # Plot evaluation metrics
            # 绘制评估指标
            plot_evaluation_metrics(evaluation_results, output_dir=output_dir, linkage=linkage)
            print(f"Recommended number of clusters: {best_n_clusters}")

    # Step 4: Perform hierarchical clustering with the final parameters
    # 步骤4：使用最终参数执行层次聚类
    if linkage == 'all' and n_clusters is None:
        # We've already performed clustering for all methods above
        # 我们已经在上面为所有方法执行了聚类
        pass
    else:
        # Perform clustering with the specified linkage method and number of clusters
        # 使用指定的连接方法和簇数执行聚类
        active_n_clusters = n_clusters if n_clusters is not None else best_n_clusters
        print(
            f"\n===== Performing hierarchical clustering with {linkage} linkage and {active_n_clusters} clusters =====")
        model = hierarchical_clustering(dtw_matrix, n_clusters=active_n_clusters, linkage=linkage)
        labels = model.labels_
        silhouette = evaluate_clustering(dtw_matrix, labels)
        print(f"Hierarchical Clustering Silhouette Score: {silhouette:.4f}")

        # Plot dendrogram (unclustered, for all hierarchical structure)
        # 绘制树状图（未聚类，显示所有层次结构）
        print("Creating model for dendrogram...")
        unclustered_model = hierarchical_clustering(
            dtw_matrix,
            n_clusters=None,
            linkage=linkage,
            distance_threshold=0.0
        )
        plot_dendrogram(unclustered_model, distance_matrix=dtw_matrix, signal_names=signal_names,
                        output_dir=output_dir, linkage=linkage)

        final_labels = labels
        final_method = f"Hierarchical Clustering with {linkage} linkage"
        final_silhouette = silhouette

    # Step 5: Visualize final clustering results
    # 步骤5：可视化最终聚类结果
    print("\n===== Visualizing final clustering results =====")

    if linkage != 'all' or n_clusters is not None:
        # Visualizations already created for 'all' linkage methods (when not using manual n_clusters)
        # 'all'连接方法的可视化已经创建（当不使用手动指定的簇数时）
        print("Generating PCA visualization...")
        plot_dimensionality_reduction(dtw_matrix, final_labels, method='pca',
                                      output_dir=output_dir, linkage=linkage)

        print("Generating t-SNE visualization...")
        plot_dimensionality_reduction(dtw_matrix, final_labels, method='tsne',
                                      output_dir=output_dir, linkage=linkage)

        # Plot cluster signals
        # 绘制簇信号
        print("Generating cluster signal plots...")
        plot_cluster_signals(time_points, preprocessed_signals, final_labels,
                             output_dir=output_dir, linkage=linkage)

    # Step 6: Print final results
    # 步骤6：打印最终结果
    print(f"\n===== Final clustering results ({final_method}) =====")
    if linkage != 'all' or n_clusters is not None:
        # Results already printed and saved for 'all' linkage methods (when not using manual n_clusters)
        # 'all'连接方法的结果已经打印和保存（当不使用手动指定的簇数时）
        print_clustering_summary(final_labels, signal_names)
        save_results(final_labels, signal_names, output_dir=output_dir, linkage=linkage)

    # Print information about how the clustering was performed
    # 打印关于聚类如何执行的信息
    if n_clusters is not None:
        print(f"\nClustering was performed with manually specified {n_clusters} clusters.")
    else:
        print(f"\nClustering was performed with automatically determined {best_n_clusters} clusters.")

    print(f"\n===== Processing completed for file: {file_path} =====")
    print(f"Results and visualizations have been saved to the directory: {output_dir}")

    return final_method, final_silhouette


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
    parser.add_argument('--file', type=str, help='Path to CSV file (for single file mode)')

    # Batch processing parameters
    # 批处理参数
    parser.add_argument('--batch', action='store_true', help='Enable batch processing mode')
    parser.add_argument('--input_dir', type=str, help='Directory containing CSV files (for batch mode)')

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
    # Add new parameter for manual cluster number specification
    # 添加新参数，用于手动指定簇的数量
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Manually specify the number of clusters (overrides automatic calculation)')

    args = parser.parse_args()

    # Validate input arguments
    # 验证输入参数
    if args.batch:
        # Batch mode - validate directory and linkage
        # 批处理模式 - 验证目录和连接方法
        if not args.input_dir:
            print("Error: --input_dir is required in batch mode")
            sys.exit(1)

        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            sys.exit(1)

        if args.linkage == 'all':
            print("Error: In batch mode, linkage cannot be 'all'. Please specify a single linkage method.")
            sys.exit(1)

        # Find all CSV files in the input directory
        # 在输入目录中查找所有CSV文件
        csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))

        if not csv_files:
            print(f"Error: No CSV files found in directory {args.input_dir}")
            sys.exit(1)

        print(f"Found {len(csv_files)} CSV files in {args.input_dir}")

        # Process each CSV file
        # 处理每个CSV文件
        for file_path in csv_files:
            # Create output directory with file name
            # 使用文件名创建输出目录
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join(args.input_dir, file_name)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'cluster_plots'), exist_ok=True)

            # Process the current file
            # 处理当前文件
            try:
                final_method, final_silhouette = process_single_file(
                    file_path=file_path,
                    sakoe_chiba_radius=args.sakoe_chiba_radius,
                    linkage=args.linkage,
                    max_clusters=args.max_clusters,
                    n_clusters=args.n_clusters,
                    output_dir=output_dir
                )

                print(
                    f"File {file_path} - Best clustering method: {final_method} (Silhouette Score: {final_silhouette:.4f})")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                print("Continuing with next file...")
                continue

        print("\n===== Batch processing completed successfully =====")

    else:
        # Single file mode
        # 单文件模式
        if not args.file:
            print("Error: --file is required in single file mode")
            sys.exit(1)

        if not os.path.isfile(args.file):
            print(f"Error: File {args.file} does not exist")
            sys.exit(1)

        # Create output directory with control options in the name
        # 创建包含控制选项的输出目录
        output_dir = ensure_output_directory(args.max_clusters, args.linkage)

        # Process the single file
        # 处理单个文件
        final_method, final_silhouette = process_single_file(
            file_path=args.file,
            sakoe_chiba_radius=args.sakoe_chiba_radius,
            linkage=args.linkage,
            max_clusters=args.max_clusters,
            n_clusters=args.n_clusters,
            output_dir=output_dir
        )

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