import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Import modules
from data_loader import load_data, preprocess_signals
from dtw_calculator import calculate_dtw_distance_matrix
from clustering import hierarchical_clustering, time_series_kmeans, dbscan_clustering
from cluster_evaluation import evaluate_clustering, find_optimal_clusters, find_optimal_dbscan_params
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
    plot_signals_overview,
    save_parameters
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
    parser.add_argument('--handle_missing', type=str, default='report',
                        choices=['report', 'drop', 'fill_mean', 'fill_zero'],
                        help='Strategy for handling missing values')

    # DTW parameters
    # DTW参数
    parser.add_argument('--sakoe_chiba_radius', type=int, default=None,
                        help='Sakoe-Chiba band radius (None = no constraint)')

    # Algorithm selection
    # 算法选择
    parser.add_argument('--algorithm', type=str, default='hierarchical',
                        choices=['hierarchical', 'kmeans', 'dbscan', 'all'],
                        help='Clustering algorithm to use')

    # Hierarchical clustering parameters
    # 层次聚类参数
    parser.add_argument('--linkage', type=str, default='average',
                        choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage criterion for hierarchical clustering')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum number of clusters to try')

    # K-means parameters
    # K均值参数
    parser.add_argument('--k_clusters', type=int, default=None,
                        help='Number of clusters for K-means (if different from hierarchical)')
    parser.add_argument('--metric', type=str, default='dtw',
                        choices=['dtw', 'softdtw'],
                        help='Distance metric for TimeSeriesKMeans')

    # DBSCAN parameters
    # DBSCAN参数
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon parameter for DBSCAN (neighborhood radius)')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum samples parameter for DBSCAN (min neighbors for core point)')

    # DBSCAN auto-parameter options
    # DBSCAN自动参数选项
    parser.add_argument('--auto_dbscan_params', action='store_true',
                        help='Automatically optimize DBSCAN parameters')
    parser.add_argument('--min_samples_range', type=int, nargs='+', default=None,
                        help='Range of min_samples values to try for DBSCAN optimization')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for parameter optimization')

    # Output options
    # 输出选项
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory path')

    args = parser.parse_args()

    # Generate timestamp for unique directory names
    # 生成时间戳用于唯一目录名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create a base directory for all results from this run
    # 为本次运行的所有结果创建一个基础目录
    base_output_dir = args.output_dir if args.output_dir else f"clustering_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Base output directory: {base_output_dir}")

    # Save input parameters for reference
    # 保存输入参数以供参考
    input_params = vars(args)
    save_parameters(input_params, base_output_dir, 'input_parameters.txt')

    # Step 1: Load and preprocess data
    # 步骤1：加载和预处理数据
    print("\n===== Step 1: Loading and preprocessing data =====")
    time_points, signal_names, signals = load_data(args.file, handle_missing=args.handle_missing)
    print(f"Loaded {len(signals)} signals with {len(time_points)} time points each")

    # Plot signals overview
    # 绘制信号概览
    plot_signals_overview(time_points, signals, signal_names, n_signals=10, output_dir=base_output_dir)

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

    # Save a copy of the distance matrix
    # 保存距离矩阵的副本
    np.save(os.path.join(base_output_dir, 'dtw_distance_matrix.npy'), dtw_matrix)
    print(f"DTW distance matrix saved to {os.path.join(base_output_dir, 'dtw_distance_matrix.npy')}")

    # Determine optimal number of clusters (only for hierarchical and kmeans)
    # 确定最优簇数量（仅用于层次聚类和K均值）
    best_n_clusters = None
    if args.algorithm in ['hierarchical', 'kmeans', 'all']:
        print("\n===== Step 3: Finding optimal number of clusters =====")
        evaluation_results, best_n_clusters = find_optimal_clusters(
            dtw_matrix,
            preprocessed_signals,
            max_clusters=args.max_clusters,
            linkage=args.linkage
        )
        # Plot evaluation metrics
        # 绘制评估指标
        plot_evaluation_metrics(evaluation_results, output_dir=base_output_dir)
        print(f"Recommended number of clusters: {best_n_clusters}")

        # Save optimal cluster number
        # 保存最优簇数
        with open(os.path.join(base_output_dir, 'optimal_clusters.txt'), 'w') as f:
            f.write(f"Recommended number of clusters: {best_n_clusters}\n")
            f.write(f"Based on evaluation metrics (silhouette score, Davies-Bouldin index, Calinski-Harabasz index)\n")

    # Step 4: Perform clustering based on selected algorithm(s)
    # 步骤4：根据选择的算法进行聚类

    # Hierarchical Clustering
    # 层次聚类
    if args.algorithm in ['hierarchical', 'all']:
        # Create algorithm-specific directory
        # 创建算法特定的目录
        hier_output_dir = ensure_output_directory('hierarchical', timestamp)

        print(f"\n===== Performing hierarchical clustering with {best_n_clusters} clusters =====")
        hier_model = hierarchical_clustering(dtw_matrix, n_clusters=best_n_clusters, linkage=args.linkage)
        hier_labels = hier_model.labels_
        hier_silhouette = evaluate_clustering(dtw_matrix, hier_labels)
        print(f"Hierarchical Clustering Silhouette Score: {hier_silhouette:.4f}")
        clustering_results['hierarchical'] = {
            'model': hier_model,
            'labels': hier_labels,
            'silhouette': hier_silhouette,
            'output_dir': hier_output_dir
        }

        # Save hierarchical clustering parameters
        # 保存层次聚类参数
        hier_params = {
            'algorithm': 'hierarchical',
            'n_clusters': best_n_clusters,
            'linkage': args.linkage,
            'silhouette_score': hier_silhouette
        }
        save_parameters(hier_params, hier_output_dir)

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
                        output_dir=hier_output_dir)

        # Plot other visualizations
        # 绘制其他可视化
        plot_dimensionality_reduction(dtw_matrix, hier_labels, method='pca', output_dir=hier_output_dir)
        plot_dimensionality_reduction(dtw_matrix, hier_labels, method='tsne', output_dir=hier_output_dir)
        plot_cluster_signals(time_points, preprocessed_signals, hier_labels, output_dir=hier_output_dir)

        # Save results
        # 保存结果
        save_results(hier_labels, signal_names, output_dir=hier_output_dir)

    # TimeSeriesKMeans
    # 时间序列K均值
    if args.algorithm in ['kmeans', 'all']:
        # Create algorithm-specific directory
        # 创建算法特定的目录
        kmeans_output_dir = ensure_output_directory('kmeans', timestamp)

        # Use specified k_clusters if provided, otherwise use optimal clusters
        # 如果提供了指定的k_clusters则使用，否则使用最优簇数
        k_clusters = args.k_clusters if args.k_clusters is not None else best_n_clusters

        print(f"\n===== Performing TimeSeriesKMeans with {k_clusters} clusters =====")
        tskm_model = time_series_kmeans(preprocessed_signals, n_clusters=k_clusters, metric=args.metric)
        tskm_labels = tskm_model.labels_
        tskm_silhouette = evaluate_clustering(dtw_matrix, tskm_labels)
        print(f"TimeSeriesKMeans Silhouette Score: {tskm_silhouette:.4f}")
        clustering_results['kmeans'] = {
            'model': tskm_model,
            'labels': tskm_labels,
            'silhouette': tskm_silhouette,
            'output_dir': kmeans_output_dir
        }

        # Save kmeans clustering parameters
        # 保存K均值聚类参数
        kmeans_params = {
            'algorithm': 'kmeans',
            'n_clusters': k_clusters,
            'metric': args.metric,
            'silhouette_score': tskm_silhouette
        }
        save_parameters(kmeans_params, kmeans_output_dir)

        # Plot visualizations
        # 绘制可视化
        plot_dimensionality_reduction(dtw_matrix, tskm_labels, method='pca', output_dir=kmeans_output_dir)
        plot_dimensionality_reduction(dtw_matrix, tskm_labels, method='tsne', output_dir=kmeans_output_dir)
        plot_cluster_signals(time_points, preprocessed_signals, tskm_labels, output_dir=kmeans_output_dir)

        # Save results
        # 保存结果
        save_results(tskm_labels, signal_names, output_dir=kmeans_output_dir)

    # DBSCAN Clustering
    # DBSCAN聚类
    if args.algorithm in ['dbscan', 'all']:
        # Create algorithm-specific directory
        # 创建算法特定的目录
        dbscan_output_dir = ensure_output_directory('dbscan', timestamp)

        # Auto-optimize DBSCAN parameters if requested
        # 如果请求，自动优化DBSCAN参数
        if args.auto_dbscan_params:
            print("\n===== Auto-optimizing DBSCAN parameters =====")
            optimal_eps, optimal_min_samples, _ = find_optimal_dbscan_params(
                dtw_matrix,
                signals=preprocessed_signals,
                min_samples_range=args.min_samples_range,
                n_jobs=args.n_jobs,
                output_dir=dbscan_output_dir
            )
            # Update the parameters
            # 更新参数
            args.eps = optimal_eps
            args.min_samples = optimal_min_samples
            print(f"Using optimized DBSCAN parameters: eps={args.eps:.4f}, min_samples={args.min_samples}")

        print(f"\n===== Performing DBSCAN clustering with eps={args.eps}, min_samples={args.min_samples} =====")
        dbscan_model = dbscan_clustering(dtw_matrix, eps=args.eps, min_samples=args.min_samples)
        dbscan_labels = dbscan_model.labels_

        # Save DBSCAN parameters
        # 保存DBSCAN参数
        dbscan_params = {
            'algorithm': 'dbscan',
            'eps': args.eps,
            'min_samples': args.min_samples,
            'auto_optimized': args.auto_dbscan_params
        }

        # Check if DBSCAN found more than one cluster (excluding noise)
        # 检查DBSCAN是否找到了多个簇（不包括噪声）
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        noise_ratio = n_noise / len(dbscan_labels) if len(dbscan_labels) > 0 else 0

        dbscan_params['n_clusters'] = n_clusters
        dbscan_params['n_noise'] = n_noise
        dbscan_params['noise_ratio'] = f"{noise_ratio:.2%}"

        if n_clusters > 0:
            dbscan_silhouette = evaluate_clustering(dtw_matrix, dbscan_labels)
            print(f"DBSCAN Clustering Silhouette Score: {dbscan_silhouette:.4f}")
            dbscan_params['silhouette_score'] = dbscan_silhouette

            clustering_results['dbscan'] = {
                'model': dbscan_model,
                'labels': dbscan_labels,
                'silhouette': dbscan_silhouette,
                'output_dir': dbscan_output_dir
            }
        else:
            print("DBSCAN did not find any meaningful clusters with current parameters")
            dbscan_params['silhouette_score'] = 'N/A (no valid clusters)'

            # If all points are noise, we cannot calculate silhouette score
            # 如果所有点都是噪声，我们无法计算轮廓系数
            clustering_results['dbscan'] = {
                'model': dbscan_model,
                'labels': dbscan_labels,
                'silhouette': -1,
                'output_dir': dbscan_output_dir
            }

        # Save DBSCAN parameters
        # 保存DBSCAN参数
        save_parameters(dbscan_params, dbscan_output_dir)

        # Plot visualizations
        # 绘制可视化
        plot_dimensionality_reduction(dtw_matrix, dbscan_labels, method='pca', output_dir=dbscan_output_dir)
        plot_dimensionality_reduction(dtw_matrix, dbscan_labels, method='tsne', output_dir=dbscan_output_dir)
        plot_cluster_signals(time_points, preprocessed_signals, dbscan_labels, output_dir=dbscan_output_dir)

        # Save results
        # 保存结果
        save_results(dbscan_labels, signal_names, output_dir=dbscan_output_dir)

    # Step 5: Determine the best clustering result
    # 步骤5：确定最佳聚类结果
    if clustering_results:
        # Compare silhouette scores and choose the best method
        # 比较轮廓系数并选择最佳方法
        valid_results = {k: v for k, v in clustering_results.items() if v['silhouette'] > -1}

        if valid_results:
            best_method = max(valid_results, key=lambda x: valid_results[x]['silhouette'])
            final_labels = valid_results[best_method]['labels']
            final_method = best_method.capitalize() + " Clustering"
            final_silhouette = valid_results[best_method]['silhouette']
            final_output_dir = valid_results[best_method]['output_dir']

            print(f"\n===== Best clustering method: {final_method} (Silhouette Score: {final_silhouette:.4f}) =====")

            # Create a symbolic link or copy the best results to a "best_method" directory
            # 创建一个符号链接或将最佳结果复制到"best_method"目录
            best_dir = ensure_output_directory('best_method', timestamp)

            # Save best method summary
            # 保存最佳方法摘要
            with open(os.path.join(best_dir, 'best_method_summary.txt'), 'w') as f:
                f.write(f"Best clustering method: {final_method}\n")
                f.write(f"Silhouette Score: {final_silhouette:.4f}\n")
                f.write(f"Number of clusters: {len(np.unique(final_labels)) - (1 if -1 in final_labels else 0)}\n")
                if -1 in final_labels:
                    noise_count = np.sum(final_labels == -1)
                    f.write(f"Number of noise points: {noise_count} ({noise_count / len(final_labels):.2%})\n")
                f.write(f"Original results directory: {final_output_dir}\n")

            # Copy the best results to the best_method directory
            # 将最佳结果复制到best_method目录
            import shutil
            for file in os.listdir(final_output_dir):
                src_file = os.path.join(final_output_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(best_dir, file))
                elif os.path.isdir(src_file):
                    shutil.copytree(src_file, os.path.join(best_dir, file), dirs_exist_ok=True)
        else:
            print("\nNo valid clustering results available. All methods produced poor quality clustering.")
    else:
        print("\nNo clustering results available. Please check your input parameters.")
        sys.exit(1)

    # Step 6: Print final summary
    # 步骤6：打印最终摘要
    print("\n===== Clustering completed successfully =====")
    print(f"Base output directory: {base_output_dir}")

    if 'hierarchical' in clustering_results and 'kmeans' in clustering_results:
        print(f"Recommended number of clusters: {best_n_clusters}")

    print("\nResults per algorithm:")
    for algorithm, result in clustering_results.items():
        if result['silhouette'] > -1:
            print(f"  {algorithm.capitalize()}: Silhouette Score = {result['silhouette']:.4f}")
        else:
            print(f"  {algorithm.capitalize()}: Invalid clustering (could not calculate silhouette score)")

    if valid_results:
        print(f"\nBest clustering method: {final_method}")
        print(f"Directory with best results: {best_dir}")

    print("\nAll results and visualizations have been saved to their respective directories.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)