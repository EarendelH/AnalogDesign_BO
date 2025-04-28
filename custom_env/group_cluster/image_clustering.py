import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skimage.feature import greycomatrix, greycoprops, hog
from skimage import color
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tabulate import tabulate
import seaborn as sns
import warnings
import time
from datetime import datetime

# 忽略警告信息
# Ignore warning messages
warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可重复
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_images(image_dir, image_type, max_images=None):
    """
    加载指定目录下的指定类型的图像
    Load images of specified type from the given directory

    Args:
        image_dir (str): 图像目录路径
                         Path to image directory
        image_type (str): 图像类型 ('wavelet', 'markov', 'multichannel', 'all')
                          Image type
        max_images (int, optional): 最大加载图像数量
                                    Maximum number of images to load

    Returns:
        tuple: (image_paths, image_names) 图像路径列表和图像名称列表
                                         List of image paths and list of image names
    """
    print(f"\nLoading {image_type} images from {image_dir}...")

    if image_type == 'all':
        # 加载所有类型的图像
        # Load all types of images
        image_paths = []
        image_names = []

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png') and not file.startswith('.'):
                    # 排除隐藏文件
                    # Exclude hidden files
                    image_paths.append(os.path.join(root, file))
                    # 提取信号名称（移除前缀和扩展名）
                    # Extract signal name (remove prefix and extension)
                    prefix = file.split('_')[0] + '_'
                    name = file.replace(prefix, '').replace('.png', '')
                    image_names.append(name)
    else:
        # 只加载特定类型的图像
        # Load only specific type of images
        subdir = os.path.join(image_dir, image_type)
        if not os.path.exists(subdir):
            print(f"Error: Directory not found: {subdir}")
            sys.exit(1)

        image_paths = []
        image_names = []

        for file in os.listdir(subdir):
            if file.endswith('.png') and not file.startswith('.'):
                image_paths.append(os.path.join(subdir, file))
                # 提取信号名称（移除前缀和扩展名）
                # Extract signal name (remove prefix and extension)
                prefix = file.split('_')[0] + '_'
                name = file.replace(prefix, '').replace('.png', '')
                image_names.append(name)

    # 如果指定了最大图像数量，则限制加载的图像数量
    # If maximum number of images is specified, limit the number of images to load
    if max_images is not None and max_images < len(image_paths):
        indices = np.random.choice(len(image_paths), max_images, replace=False)
        image_paths = [image_paths[i] for i in indices]
        image_names = [image_names[i] for i in indices]

    print(f"Loaded {len(image_paths)} images.")
    return image_paths, image_names


def extract_traditional_features(image_path):
    """
    提取传统图像特征（纹理特征、HOG特征和直方图特征）
    Extract traditional image features (texture features, HOG features, and histogram features)

    Args:
        image_path (str): 图像文件路径
                          Path to image file

    Returns:
        numpy.ndarray: 特征向量
                       Feature vector
    """
    # 读取图像
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # 转换为灰度图
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 调整图像尺寸以保持一致性
    # Resize image for consistency
    gray_resized = cv2.resize(gray, (256, 256))

    # 1. 提取GLCM纹理特征
    # 1. Extract GLCM texture features
    try:
        # 将图像值调整到[0, 255]范围
        # Adjust image values to [0, 255] range
        gray_norm = cv2.normalize(gray_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 计算GLCM矩阵（距离=1，角度=[0, 45, 90, 135]，灰度级别=8）
        # Compute GLCM matrix (distance=1, angles=[0, 45, 90, 135], gray levels=8)
        glcm = greycomatrix(gray_norm, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 8, symmetric=True, normed=True)

        # 提取GLCM属性
        # Extract GLCM properties
        contrast = greycoprops(glcm, 'contrast').flatten()
        dissimilarity = greycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = greycoprops(glcm, 'homogeneity').flatten()
        energy = greycoprops(glcm, 'energy').flatten()
        correlation = greycoprops(glcm, 'correlation').flatten()

        # 合并纹理特征
        # Combine texture features
        texture_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    except Exception as e:
        print(f"Error extracting GLCM features: {e}")
        # 如果GLCM特征提取失败，使用零向量
        # Use zero vector if GLCM feature extraction fails
        texture_features = np.zeros(20)

    # 2. 提取HOG特征
    # 2. Extract HOG features
    try:
        # 使用较小的细胞大小和块大小来减少特征维度
        # Use smaller cell size and block size to reduce feature dimension
        hog_features = hog(gray_resized, orientations=8, pixels_per_cell=(32, 32),
                           cells_per_block=(2, 2), visualize=False)
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        # 如果HOG特征提取失败，使用零向量
        # Use zero vector if HOG feature extraction fails
        hog_features = np.zeros(64)

    # 3. 提取颜色直方图特征
    # 3. Extract color histogram features
    try:
        # 计算各通道的直方图
        # Compute histogram for each channel
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)

        # 添加灰度直方图
        # Add grayscale histogram
        gray_hist = cv2.calcHist([gray_resized], [0], None, [16], [0, 256])
        gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
        hist_features.extend(gray_hist)

        hist_features = np.array(hist_features)
    except Exception as e:
        print(f"Error extracting histogram features: {e}")
        # 如果直方图特征提取失败，使用零向量
        # Use zero vector if histogram feature extraction fails
        hist_features = np.zeros(40)

    # 合并所有特征
    # Combine all features
    all_features = np.concatenate([texture_features, hog_features, hist_features])

    return all_features


def extract_deep_features(image_path, model_name='resnet18'):
    """
    使用预训练的深度学习模型提取图像特征
    Extract image features using a pre-trained deep learning model

    Args:
        image_path (str): 图像文件路径
                          Path to image file
        model_name (str, optional): 模型名称 ('resnet18', 'mobilenet_v2')
                                    Model name

    Returns:
        numpy.ndarray: 特征向量
                       Feature vector
    """
    # 定义图像预处理
    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # 加载图像
        # Load image
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)

        # 添加批次维度
        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0)

        # 设置设备（如果有GPU则使用GPU）
        # Set device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练模型
        # Load pre-trained model
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            # 移除最后的全连接层
            # Remove the last fully connected layer
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            # 移除分类器
            # Remove classifier
            model = model.features
        else:
            print(f"Error: Unsupported model {model_name}")
            return None

        # 将模型设置为评估模式
        # Set model to evaluation mode
        model.eval()
        model.to(device)

        # 将输入数据移动到相同设备
        # Move input data to the same device
        input_batch = input_batch.to(device)

        # 关闭梯度计算以加速推理
        # Turn off gradient computation for inference
        with torch.no_grad():
            # 前向传播
            # Forward pass
            features = model(input_batch)

        # 将特征从GPU移动到CPU（如果适用），并转换为NumPy数组
        # Move features from GPU to CPU (if applicable) and convert to NumPy array
        features = features.cpu().numpy()

        # 压缩维度并返回特征向量
        # Squeeze dimensions and return feature vector
        return features.squeeze()

    except Exception as e:
        print(f"Error extracting deep features from {image_path}: {e}")
        return None


def kmeans_clustering(features, n_clusters=5):
    """
    使用K-means算法对特征进行聚类
    Cluster features using K-means algorithm

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        n_clusters (int, optional): 簇的数量
                                    Number of clusters

    Returns:
        tuple: (labels, metrics) 聚类标签和评估指标
                                 Clustering labels and evaluation metrics
    """
    # 创建并拟合K-means模型
    # Create and fit K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # 计算评估指标
    # Calculate evaluation metrics
    metrics = calculate_metrics(features, labels)

    return labels, metrics


def dbscan_clustering(features, eps=0.5, min_samples=5):
    """
    使用DBSCAN算法对特征进行聚类
    Cluster features using DBSCAN algorithm

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        eps (float, optional): 邻域半径
                               Neighborhood radius
        min_samples (int, optional): 成为核心点所需的最小样本数
                                     Minimum number of samples required to be a core point

    Returns:
        tuple: (labels, metrics) 聚类标签和评估指标
                                 Clustering labels and evaluation metrics
    """
    # 创建并拟合DBSCAN模型
    # Create and fit DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    # 计算评估指标（当存在不止一个簇且不是所有点都是噪声点时）
    # Calculate evaluation metrics (when there is more than one cluster and not all points are noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1 and not all(l == -1 for l in labels):
        metrics = calculate_metrics(features, labels)
    else:
        # 如果所有点都被归为一个簇或噪声点，则评估指标为空
        # If all points are classified as one cluster or noise, evaluation metrics are empty
        metrics = {'silhouette': float('nan'), 'calinski_harabasz': float('nan'), 'davies_bouldin': float('nan')}

    return labels, metrics


def hierarchical_clustering(features, n_clusters=5):
    """
    使用层次聚类算法对特征进行聚类
    Cluster features using hierarchical clustering algorithm

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        n_clusters (int, optional): 簇的数量
                                    Number of clusters

    Returns:
        tuple: (labels, metrics) 聚类标签和评估指标
                                 Clustering labels and evaluation metrics
    """
    # 创建并拟合层次聚类模型
    # Create and fit hierarchical clustering model
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hc.fit_predict(features)

    # 计算评估指标
    # Calculate evaluation metrics
    metrics = calculate_metrics(features, labels)

    return labels, metrics


def calculate_metrics(features, labels):
    """
    计算聚类评估指标
    Calculate clustering evaluation metrics

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        labels (numpy.ndarray): 聚类标签
                                Clustering labels

    Returns:
        dict: 评估指标字典
              Dictionary of evaluation metrics
    """
    metrics = {}

    # 获取非噪声点的索引（对于DBSCAN可能会有噪声点）
    # Get indices of non-noise points (DBSCAN may have noise points)
    non_noise_idx = labels != -1

    # 如果所有点都是噪声点，返回NaN值
    # If all points are noise points, return NaN values
    if not np.any(non_noise_idx):
        metrics['silhouette'] = float('nan')
        metrics['calinski_harabasz'] = float('nan')
        metrics['davies_bouldin'] = float('nan')
        return metrics

    # 过滤噪声点
    # Filter out noise points
    features_filtered = features[non_noise_idx]
    labels_filtered = labels[non_noise_idx]

    # 如果只有一个簇，轮廓系数无法计算
    # If there is only one cluster, silhouette score cannot be calculated
    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        metrics['silhouette'] = float('nan')
        metrics['calinski_harabasz'] = float('nan')
        metrics['davies_bouldin'] = float('nan')
        return metrics

    try:
        # 计算轮廓系数
        # Calculate silhouette score
        metrics['silhouette'] = silhouette_score(features_filtered, labels_filtered)
    except:
        metrics['silhouette'] = float('nan')

    try:
        # 计算Calinski-Harabasz指数
        # Calculate Calinski-Harabasz index
        metrics['calinski_harabasz'] = calinski_harabasz_score(features_filtered, labels_filtered)
    except:
        metrics['calinski_harabasz'] = float('nan')

    try:
        # 计算Davies-Bouldin指数
        # Calculate Davies-Bouldin index
        metrics['davies_bouldin'] = davies_bouldin_score(features_filtered, labels_filtered)
    except:
        metrics['davies_bouldin'] = float('nan')

    return metrics


def reduce_dimensions(features, n_components=2, method='tsne'):
    """
    对特征进行降维
    Reduce dimensions of features

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        n_components (int, optional): 降维后的维度
                                     Number of dimensions after reduction
        method (str, optional): 降维方法 ('pca', 'tsne')
                                Dimensionality reduction method

    Returns:
        numpy.ndarray: 降维后的特征
                       Reduced features
    """
    if method == 'pca':
        # 使用PCA进行降维
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
    elif method == 'tsne':
        # 使用t-SNE进行降维
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_features = tsne.fit_transform(features)
    else:
        print(f"Error: Unsupported dimensionality reduction method {method}")
        reduced_features = None

    return reduced_features


def visualize_clustering(features, labels, image_names, title, output_path):
    """
    可视化聚类结果并保存图像
    Visualize clustering results and save image

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        labels (numpy.ndarray): 聚类标签
                                Clustering labels
        image_names (list): 图像名称列表
                            List of image names
        title (str): 可视化标题
                     Visualization title
        output_path (str): 输出路径
                           Output path
    """
    # 如果特征维度大于2，则使用t-SNE进行降维
    # If feature dimension is greater than 2, use t-SNE for dimensionality reduction
    if features.shape[1] > 2:
        reduced_features = reduce_dimensions(features, method='tsne')
    else:
        reduced_features = features

    # 创建可视化
    # Create visualization
    plt.figure(figsize=(12, 10))

    # 获取聚类的唯一标签
    # Get unique cluster labels
    unique_labels = np.unique(labels)

    # 为每个聚类设置不同的颜色
    # Set different colors for each cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # 绘制各个聚类
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点用黑色表示
            # Noise points are represented in black
            cluster_color = 'k'
            marker = 'x'
        else:
            cluster_color = colors[i]
            marker = 'o'

        # 获取属于该聚类的点的索引
        # Get indices of points belonging to this cluster
        idx = labels == label

        # 绘制该聚类的点
        # Plot points of this cluster
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1],
                    color=cluster_color, marker=marker, s=50, label=f'Cluster {label}')

    # 添加标题和轴标签
    # Add title and axis labels
    plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)

    # 添加图例
    # Add legend
    plt.legend(fontsize=10)

    # 调整布局
    # Adjust layout
    plt.tight_layout()

    # 保存图像
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Clustering visualization saved to {output_path}")

    # 创建详细的聚类报告
    # Create detailed clustering report
    create_cluster_report(labels, image_names, title, output_path.replace('.png', '_report.csv'))


def create_cluster_report(labels, image_names, title, output_path):
    """
    创建聚类报告并保存为CSV文件
    Create clustering report and save as CSV file

    Args:
        labels (numpy.ndarray): 聚类标签
                                Clustering labels
        image_names (list): 图像名称列表
                            List of image names
        title (str): 报告标题
                     Report title
        output_path (str): 输出路径
                           Output path
    """
    # 创建数据框
    # Create dataframe
    df = pd.DataFrame({
        'image_name': image_names,
        'cluster': labels
    })

    # 按簇排序
    # Sort by cluster
    df = df.sort_values('cluster')

    # 保存为CSV文件
    # Save as CSV file
    df.to_csv(output_path, index=False)
    print(f"Clustering report saved to {output_path}")


def generate_results_table(results, output_dir):
    """
    生成结果比较表格并保存
    Generate results comparison table and save

    Args:
        results (dict): 结果字典
                        Dictionary of results
        output_dir (str): 输出目录
                          Output directory
    """
    # 创建表格数据
    # Create table data
    table_data = []

    # 定义评估指标
    # Define evaluation metrics
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

    # 对于每种图像类型
    # For each image type
    for img_type in ['wavelet', 'markov', 'multichannel']:
        # 对于每种特征提取方法
        # For each feature extraction method
        for feature_type in ['traditional', 'deep']:
            # 对于每种聚类算法
            # For each clustering algorithm
            for cluster_method in ['kmeans', 'dbscan', 'hierarchical']:
                # 获取结果键
                # Get result key
                key = f"{img_type}_{feature_type}_{cluster_method}"

                # 如果键存在
                # If key exists
                if key in results:
                    # 获取评估指标
                    # Get evaluation metrics
                    metric_values = results[key]

                    # 添加到表格数据
                    # Add to table data
                    row = [img_type, feature_type, cluster_method]
                    for metric in metrics:
                        row.append(f"{metric_values[metric]:.4f}" if not np.isnan(metric_values[metric]) else "N/A")

                    table_data.append(row)

    # 创建数据框
    # Create dataframe
    df = pd.DataFrame(table_data, columns=['Image Type', 'Feature Type', 'Clustering Method',
                                           'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score'])

    # 保存为CSV文件
    # Save as CSV file
    csv_path = os.path.join(output_dir, 'clustering_results_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults comparison table saved to {csv_path}")

    # 创建热力图
    # Create heatmap
    create_metrics_heatmap(df, output_dir)


def create_metrics_heatmap(df, output_dir):
    """
    创建评估指标热力图
    Create heatmap of evaluation metrics

    Args:
        df (pandas.DataFrame): 结果数据框
                               Results dataframe
        output_dir (str): 输出目录
                          Output directory
    """
    # 对于每个指标创建热力图
    # Create heatmap for each metric
    metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']

    for metric in metrics:
        # 创建用于热力图的数据框
        # Create dataframe for heatmap
        heatmap_df = df.pivot_table(
            index=['Image Type', 'Feature Type'],
            columns='Clustering Method',
            values=metric,
            aggfunc='mean'
        )

        # 对于Davies-Bouldin指数，较小的值更好
        # For Davies-Bouldin index, smaller values are better
        if metric == 'Davies-Bouldin Score':
            # 将"N/A"转换为NaN
            # Convert "N/A" to NaN
            heatmap_df = heatmap_df.applymap(lambda x: np.nan if x == "N/A" else float(x))
            # 反转颜色映射（较小的值为红色）
            # Reverse colormap (smaller values are red)
            cmap = 'RdYlGn_r'
        else:
            # 将"N/A"转换为NaN
            # Convert "N/A" to NaN
            heatmap_df = heatmap_df.applymap(lambda x: np.nan if x == "N/A" else float(x))
            # 使用标准颜色映射（较大的值为绿色）
            # Use standard colormap (larger values are green)
            cmap = 'RdYlGn'

        # 创建热力图
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, cmap=cmap, fmt='.4f', linewidths=0.5)

        # 添加标题
        # Add title
        plt.title(f'Comparison of {metric} Across Methods', fontsize=16)

        # 调整布局
        # Adjust layout
        plt.tight_layout()

        # 保存热力图
        # Save heatmap
        metric_filename = metric.lower().replace(' ', '_').replace('-', '_')
        output_path = os.path.join(output_dir, f'heatmap_{metric_filename}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Heatmap for {metric} saved to {output_path}")


def run_clustering(image_paths, image_names, feature_type, cluster_method, n_clusters=5, output_dir=None,
                   eps=0.5, min_samples=5, visualization_title=None):
    """
    运行完整的聚类流程
    Run complete clustering pipeline

    Args:
        image_paths (list): 图像路径列表
                            List of image paths
        image_names (list): 图像名称列表
                            List of image names
        feature_type (str): 特征类型 ('traditional', 'deep')
                            Feature type
        cluster_method (str): 聚类方法 ('kmeans', 'dbscan', 'hierarchical')
                              Clustering method
        n_clusters (int, optional): 簇的数量（用于K-means和层次聚类）
                                    Number of clusters (for K-means and hierarchical clustering)
        output_dir (str, optional): 输出目录
                                    Output directory
        eps (float, optional): DBSCAN的邻域半径
                               Neighborhood radius for DBSCAN
        min_samples (int, optional): DBSCAN的最小样本数
                                     Minimum number of samples for DBSCAN
        visualization_title (str, optional): 可视化标题
                                             Visualization title

    Returns:
        tuple: (labels, metrics, features) 聚类标签、评估指标和特征
                                           Clustering labels, evaluation metrics, and features
    """
    # 设置默认输出目录
    # Set default output directory
    if output_dir is None:
        output_dir = 'clustering_results'
    os.makedirs(output_dir, exist_ok=True)

    # 设置默认可视化标题
    # Set default visualization title
    if visualization_title is None:
        visualization_title = f"{feature_type.capitalize()} Features + {cluster_method.capitalize()} Clustering"

    # 提取特征
    # Extract features
    print(f"\nExtracting {feature_type} features from {len(image_paths)} images...")
    features_list = []

    for image_path in tqdm(image_paths):
        if feature_type == 'traditional':
            features = extract_traditional_features(image_path)
        elif feature_type == 'deep':
            features = extract_deep_features(image_path)
        else:
            print(f"Error: Unsupported feature type {feature_type}")
            return None, None, None

        if features is not None:
            features_list.append(features)

    # 检查是否所有特征都成功提取
    # Check if all features were successfully extracted
    if len(features_list) != len(image_paths):
        print(f"Warning: Only {len(features_list)} out of {len(image_paths)} features were extracted successfully.")

        # 更新图像名称列表以匹配成功提取的特征
        # Update image names list to match successfully extracted features
        valid_indices = [i for i, f in enumerate(features_list) if f is not None]
        image_names = [image_names[i] for i in valid_indices if i < len(image_names)]

    # 将特征列表转换为NumPy数组
    # Convert features list to NumPy array
    features_array = np.array(features_list)

    # 标准化特征
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # 聚类
    # Clustering
    print(f"\nPerforming {cluster_method} clustering...")
    if cluster_method == 'kmeans':
        labels, metrics = kmeans_clustering(features_scaled, n_clusters=n_clusters)
    elif cluster_method == 'dbscan':
        labels, metrics = dbscan_clustering(features_scaled, eps=eps, min_samples=min_samples)
    elif cluster_method == 'hierarchical':
        labels, metrics = hierarchical_clustering(features_scaled, n_clusters=n_clusters)
    else:
        print(f"Error: Unsupported clustering method {cluster_method}")
        return None, None, None

    # 输出评估指标
    # Output evaluation metrics
    print("\nClustering evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        if not np.isnan(metric_value):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: N/A")

    # 获取聚类的唯一标签和每个聚类的大小
    # Get unique cluster labels and size of each cluster
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]

    print("\nClustering results:")
    for label, size in zip(unique_labels, cluster_sizes):
        if label == -1:
            print(f"  Noise points: {size}")
        else:
            print(f"  Cluster {label}: {size} images")

    # 可视化聚类结果
    # Visualize clustering results
    if output_dir is not None:
        output_path = os.path.join(output_dir, f"{feature_type}_{cluster_method}_clustering.png")
        visualize_clustering(features_scaled, labels, image_names, visualization_title, output_path)

    return labels, metrics, features_scaled


def run_all_methods(image_dir, output_dir, n_clusters=5, eps=0.5, min_samples=5, max_images=None):
    """
    运行所有聚类方法的组合
    Run all combinations of clustering methods

    Args:
        image_dir (str): 图像目录
                         Image directory
        output_dir (str): 输出目录
                          Output directory
        n_clusters (int, optional): 簇的数量（用于K-means和层次聚类）
                                    Number of clusters (for K-means and hierarchical clustering)
        eps (float, optional): DBSCAN的邻域半径
                               Neighborhood radius for DBSCAN
        min_samples (int, optional): DBSCAN的最小样本数
                                     Minimum number of samples for DBSCAN
        max_images (int, optional): 每种类型的最大图像数量
                                    Maximum number of images for each type
    """
    # 创建输出目录
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 定义图像类型、特征类型和聚类方法
    # Define image types, feature types, and clustering methods
    image_types = ['wavelet', 'markov', 'multichannel']
    feature_types = ['traditional', 'deep']
    cluster_methods = ['kmeans', 'dbscan', 'hierarchical']

    # 存储所有结果
    # Store all results
    all_results = {}

    # 对于每种图像类型
    # For each image type
    for img_type in image_types:
        print(f"\n\n===== Processing {img_type} images =====")

        # 加载图像
        # Load images
        img_dir = os.path.join(image_dir, img_type)
        if not os.path.exists(img_dir):
            print(f"Error: Directory not found: {img_dir}")
            continue

        image_paths, image_names = load_images(image_dir, img_type, max_images=max_images)

        if len(image_paths) == 0:
            print(f"No images found in {img_dir}")
            continue

        # 创建该图像类型的输出目录
        # Create output directory for this image type
        img_output_dir = os.path.join(output_dir, img_type)
        os.makedirs(img_output_dir, exist_ok=True)

        # 对于每种特征类型
        # For each feature type
        for feature_type in feature_types:
            # 对于每种聚类方法
            # For each clustering method
            for cluster_method in cluster_methods:
                print(f"\n----- {img_type} + {feature_type} features + {cluster_method} clustering -----")

                # 设置可视化标题
                # Set visualization title
                visualization_title = f"{img_type.capitalize()} + {feature_type.capitalize()} Features + {cluster_method.capitalize()} Clustering"

                # 运行聚类
                # Run clustering
                _, metrics, _ = run_clustering(
                    image_paths, image_names, feature_type, cluster_method,
                    n_clusters=n_clusters, output_dir=img_output_dir,
                    eps=eps, min_samples=min_samples,
                    visualization_title=visualization_title
                )

                # 存储结果
                # Store results
                result_key = f"{img_type}_{feature_type}_{cluster_method}"
                all_results[result_key] = metrics

    # 生成结果比较表格
    # Generate results comparison table
    generate_results_table(all_results, output_dir)


def main():
    """
    主函数
    Main function
    """
    # 解析命令行参数
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Time Series Image Clustering')

    # 数据输入参数
    # Data input parameters
    parser.add_argument('--image_dir', type=str, default='image_representations',
                        help='Directory containing image representations')
    parser.add_argument('--image_type', type=str, default='wavelet',
                        choices=['wavelet', 'markov', 'multichannel', 'all'],
                        help='Type of image representation to use')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')

    # 聚类方法参数
    # Clustering method parameters
    parser.add_argument('--method', type=str, default='traditional_kmeans',
                        choices=['traditional_kmeans', 'traditional_dbscan', 'traditional_hierarchical',
                                 'deep_kmeans', 'deep_dbscan', 'deep_hierarchical', 'run_all'],
                        help='Clustering method to use')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for K-means and hierarchical clustering')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum samples parameter for DBSCAN')

    # 输出参数
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='clustering_results',
                        help='Directory to save clustering results')

    args = parser.parse_args()

    # 创建输出目录
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置开始时间
    # Set start time
    start_time = time.time()

    # 运行指定的方法
    # Run specified method
    if args.method == 'run_all':
        print("\n===== Running all clustering methods =====")
        run_all_methods(args.image_dir, args.output_dir,
                        n_clusters=args.n_clusters,
                        eps=args.eps, min_samples=args.min_samples,
                        max_images=args.max_images)
    else:
        # 分解方法名称
        # Decompose method name
        feature_type, cluster_method = args.method.split('_')

        # 加载图像
        # Load images
        image_paths, image_names = load_images(args.image_dir, args.image_type, args.max_images)

        if len(image_paths) == 0:
            print(f"Error: No images found in {args.image_dir} for type {args.image_type}")
            sys.exit(1)

        # 设置可视化标题
        # Set visualization title
        visualization_title = f"{args.image_type.capitalize()} + {feature_type.capitalize()} Features + {cluster_method.capitalize()} Clustering"

        # 运行聚类
        # Run clustering
        run_clustering(image_paths, image_names, feature_type, cluster_method,
                       n_clusters=args.n_clusters, output_dir=args.output_dir,
                       eps=args.eps, min_samples=args.min_samples,
                       visualization_title=visualization_title)

    # 计算运行时间
    # Calculate run time
    end_time = time.time()
    run_time = end_time - start_time

    print(f"\n===== Clustering completed in {run_time:.2f} seconds =====")
    print(f"Results have been saved to the '{args.output_dir}' directory")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)