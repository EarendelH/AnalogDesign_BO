import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skimage.feature import graycomatrix, graycoprops, hog
from skimage import color
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
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

    # 只处理特定类型的图像（不再允许'all'直接加载所有图像）
    # Only handle specific image types (no longer allow 'all' to load all images directly)
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

        # 计算GLCM矩阵（距离=1，角度=[0, 45, 90, 135]，灰度级别=256）
        # Compute GLCM matrix (distance=1, angles=[0, 45, 90, 135], gray levels=256)
        glcm = graycomatrix(gray_norm, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)

        # 提取GLCM属性
        # Extract GLCM properties
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()

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


def determine_optimal_clusters(features, max_clusters=15, image_type=None, feature_type=None, output_dir=None):
    """
    自动确定最佳聚类数量
    Automatically determine the optimal number of clusters

    Args:
        features (numpy.ndarray): 特征矩阵
                                  Feature matrix
        max_clusters (int, optional): 要尝试的最大聚类数量
                                      Maximum number of clusters to try
        image_type (str, optional): 图像类型，用于文件名
                                   Image type for filename
        feature_type (str, optional): 特征类型，用于文件名
                                     Feature type for filename
        output_dir (str, optional): 输出目录
                                   Output directory

    Returns:
        int: 最佳聚类数量
             Optimal number of clusters
    """
    print("\nDetermining optimal number of clusters...")

    # 确保最大聚类数不超过样本数的一半
    # Ensure max clusters does not exceed half the number of samples
    max_clusters = min(max_clusters, features.shape[0] // 2)

    from sklearn.cluster import KMeans

    # 初始化评分列表
    # Initialize score lists
    inertia_values = []
    silhouette_values = []
    calinski_values = []

    # 尝试不同的聚类数量
    # Try different numbers of clusters
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in tqdm(cluster_range):
        # K-means聚类
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # 计算惯性（簇内平方和）
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)

        # 计算轮廓系数
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(features, labels)
            silhouette_values.append(silhouette_avg)
        except:
            silhouette_values.append(0)

        # 计算Calinski-Harabasz指数
        # Calculate Calinski-Harabasz index
        try:
            calinski = calinski_harabasz_score(features, labels)
            calinski_values.append(calinski)
        except:
            calinski_values.append(0)

    # 计算肘部法则的结果
    # Calculate elbow method result
    elbow_optimal = find_elbow_point(cluster_range, inertia_values)

    # 找到轮廓系数最大的聚类数量
    # Find number of clusters with maximum silhouette score
    silhouette_optimal = cluster_range[np.argmax(silhouette_values)] if silhouette_values else elbow_optimal

    # 找到Calinski-Harabasz指数最大的聚类数量
    # Find number of clusters with maximum Calinski-Harabasz index
    calinski_optimal = cluster_range[np.argmax(calinski_values)] if calinski_values else elbow_optimal

    # 构建唯一的输出文件名
    # Build unique output filename
    filename_prefix = ""
    if image_type:
        filename_prefix += f"{image_type}_"
    if feature_type:
        filename_prefix += f"{feature_type}_"

    output_path = 'optimal_clusters_evaluation.png'
    if output_dir:
        if filename_prefix:
            output_path = os.path.join(output_dir, f"{filename_prefix}optimal_clusters_evaluation.png")
        else:
            output_path = os.path.join(output_dir, output_path)

    # 绘制评估结果
    # Plot evaluation results
    plt.figure(figsize=(15, 10))

    # 绘制惯性曲线（肘部法则）
    # Plot inertia curve (elbow method)
    plt.subplot(2, 2, 1)
    plt.plot(cluster_range, inertia_values, 'bo-')
    plt.axvline(x=elbow_optimal, color='r', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method (Optimal: {elbow_optimal})')
    plt.grid(True)

    # 绘制轮廓系数曲线
    # Plot silhouette score curve
    plt.subplot(2, 2, 2)
    plt.plot(cluster_range, silhouette_values, 'go-')
    plt.axvline(x=silhouette_optimal, color='r', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Method (Optimal: {silhouette_optimal})')
    plt.grid(True)

    # 绘制Calinski-Harabasz指数曲线
    # Plot Calinski-Harabasz index curve
    plt.subplot(2, 2, 3)
    plt.plot(cluster_range, calinski_values, 'mo-')
    plt.axvline(x=calinski_optimal, color='r', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title(f'Calinski-Harabasz Method (Optimal: {calinski_optimal})')
    plt.grid(True)

    # 综合评估结果
    # Combined evaluation results
    plt.subplot(2, 2, 4)
    plt.bar(['Elbow', 'Silhouette', 'Calinski-Harabasz'], [elbow_optimal, silhouette_optimal, calinski_optimal])
    plt.ylabel('Optimal Number of Clusters')
    plt.title('Comparison of Methods')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 综合三种方法的结果，取平均值并四舍五入
    # Combine results from three methods, take average and round
    optimal_clusters = int(round((elbow_optimal + silhouette_optimal + calinski_optimal) / 3))

    print(f"\nEstimated optimal number of clusters:")
    print(f"  Elbow Method: {elbow_optimal}")
    print(f"  Silhouette Method: {silhouette_optimal}")
    print(f"  Calinski-Harabasz Method: {calinski_optimal}")
    print(f"  Recommended number of clusters: {optimal_clusters}")

    return optimal_clusters


def find_elbow_point(x, y):
    """
    根据肘部法则找到最佳拐点
    Find the optimal elbow point

    Args:
        x (list): x坐标值
                 x-coordinate values
        y (list): y坐标值
                 y-coordinate values

    Returns:
        int: 最佳拐点对应的x值
             x-value of the optimal elbow point
    """
    # 标准化x和y
    # Normalize x and y
    x_norm = np.array(x)
    y_norm = np.array(y)

    # 如果数据点太少，返回中间点
    # If too few data points, return the middle point
    if len(x_norm) < 3:
        return x_norm[len(x_norm) // 2]

    # 标准化到[0,1]范围
    # Normalize to [0,1] range
    x_norm = (x_norm - min(x_norm)) / (max(x_norm) - min(x_norm))
    y_norm = (y_norm - min(y_norm)) / (max(y_norm) - min(y_norm))

    # 计算到直线的距离
    # Calculate distance to the line
    a = np.array([x_norm[0], y_norm[0]])
    b = np.array([x_norm[-1], y_norm[-1]])

    # 计算直线的方向向量
    # Calculate direction vector of the line
    direction = b - a

    # 计算每个点到直线的距离
    # Calculate distance of each point to the line
    distances = []
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        # 垂直距离公式：|(p-a)×(b-a)|/|b-a|
        # Perpendicular distance formula: |(p-a)×(b-a)|/|b-a|
        distance = abs(np.cross(point - a, direction)) / np.linalg.norm(direction)
        distances.append(distance)

    # 找到距离最大的点
    # Find the point with maximum distance
    elbow_index = np.argmax(distances)

    return x[elbow_index]


def generate_results_table(results, output_dir, run_id=None):
    """
    生成结果比较表格并保存
    Generate results comparison table and save

    Args:
        results (dict): 结果字典
                        Dictionary of results
        output_dir (str): 输出目录
                          Output directory
        run_id (str, optional): 运行标识，用于唯一文件名
                               Run identifier for unique filenames
    """
    # 创建表格数据
    # Create table data
    table_data = []

    # 生成运行标识（如果未提供）
    # Generate run identifier (if not provided)
    suffix = ""
    if run_id:
        suffix = f"_{run_id}"
    else:
        from datetime import datetime
        suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 定义评估指标
    # Define evaluation metrics
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

    # 对于每种图像类型
    # For each image type
    for img_type in ['wavelet', 'markov', 'multichannel']:
        # 对于每种特征提取方法
        # For each feature extraction method
        for feature_type in ['traditional', 'deep']:
            # 获取结果键
            # Get result key
            key = f"{img_type}_{feature_type}_hierarchical"

            # 如果键存在
            # If key exists
            if key in results:
                # 获取评估指标
                # Get evaluation metrics
                metric_values = results[key]

                # 添加到表格数据
                # Add to table data
                row = [img_type, feature_type, 'hierarchical']
                for metric in metrics:
                    row.append(f"{metric_values[metric]:.4f}" if not np.isnan(metric_values[metric]) else "N/A")

                table_data.append(row)

    # 创建数据框
    # Create dataframe
    df = pd.DataFrame(table_data, columns=['Image Type', 'Feature Type', 'Clustering Method',
                                           'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score'])

    # 保存为CSV文件
    # Save as CSV file
    csv_path = os.path.join(output_dir, f'clustering_results_comparison{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults comparison table saved to {csv_path}")

    # 创建热力图
    # Create heatmap
    create_metrics_heatmap(df, output_dir, run_id=run_id)


def create_metrics_heatmap(df, output_dir, run_id=None):
    """
    创建评估指标热力图
    Create heatmap of evaluation metrics

    Args:
        df (pandas.DataFrame): 结果数据框
                               Results dataframe
        output_dir (str): 输出目录
                          Output directory
        run_id (str, optional): 运行标识，用于唯一文件名
                               Run identifier for unique filenames
    """
    # 对于每个指标创建热力图
    # Create heatmap for each metric
    metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']

    # 首先将所有 "N/A" 转换为 np.nan
    # First convert all "N/A" to np.nan
    df_numeric = df.copy()
    for metric in metrics:
        df_numeric[metric] = df_numeric[metric].apply(lambda x: np.nan if x == "N/A" else float(x))

    # 生成运行标识
    # Generate run identifier
    suffix = ""
    if run_id:
        suffix = f"_{run_id}"
    else:
        from datetime import datetime
        suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for metric in metrics:
        try:
            # 创建用于热力图的数据框
            # Create dataframe for heatmap
            # 使用mean作为聚合函数，忽略NaN值
            # Use mean as aggregation function, ignore NaN values
            heatmap_df = pd.pivot_table(
                df_numeric,
                index=['Image Type'],
                columns='Feature Type',
                values=metric,
                aggfunc='mean'
            )

            # 对于Davies-Bouldin指数，较小的值更好
            # For Davies-Bouldin index, smaller values are better
            if metric == 'Davies-Bouldin Score':
                # 使用反转的颜色映射（较小的值为绿色）
                # Use reversed colormap (smaller values are green)
                cmap = 'RdYlGn_r'
            else:
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
            output_path = os.path.join(output_dir, f'heatmap_{metric_filename}{suffix}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Heatmap for {metric} saved to {output_path}")
        except Exception as e:
            print(f"Warning: Could not create heatmap for {metric}: {str(e)}")


def run_clustering(image_paths, image_names, feature_type, output_dir=None,
                   visualization_title=None, auto_clusters=False, n_clusters=5,
                   image_type=None, run_id=None):
    """
    运行完整的层次聚类流程
    Run complete hierarchical clustering pipeline

    Args:
        image_paths (list): 图像路径列表
                            List of image paths
        image_names (list): 图像名称列表
                            List of image names
        feature_type (str): 特征类型 ('traditional', 'deep')
                            Feature type
        output_dir (str, optional): 输出目录
                                    Output directory
        visualization_title (str, optional): 可视化标题
                                             Visualization title
        auto_clusters (bool, optional): 是否自动确定最佳聚类数量
                                        Whether to automatically determine the optimal number of clusters
        n_clusters (int, optional): 簇的数量（用于层次聚类）
                                    Number of clusters (for hierarchical clustering)
        image_type (str, optional): 图像类型，用于文件名
                                   Image type for filename
        run_id (str, optional): 运行标识，用于唯一文件名
                               Run identifier for unique filenames

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
        visualization_title = f"{feature_type.capitalize()} Features + Hierarchical Clustering"

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

    # 如果启用了自动确定聚类数量
    # If auto_clusters is enabled
    if auto_clusters:
        optimal_n_clusters = determine_optimal_clusters(
            features_scaled,
            image_type=image_type,
            feature_type=feature_type,
            output_dir=output_dir
        )
        print(f"Using automatically determined number of clusters: {optimal_n_clusters}")
        n_clusters = optimal_n_clusters

    # 层次聚类
    # Hierarchical clustering
    print(f"\nPerforming hierarchical clustering...")
    labels, metrics = hierarchical_clustering(features_scaled, n_clusters=n_clusters)

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
        print(f"  Cluster {label}: {size} images")

    # 可视化聚类结果
    # Visualize clustering results
    if output_dir is not None:
        # 构建文件名前缀
        filename_prefix = ""
        if image_type:
            filename_prefix += f"{image_type}_"
        filename_prefix += f"{feature_type}_hierarchical"

        # 添加聚类数量或自动聚类标志
        if auto_clusters:
            filename_prefix += f"_auto"
        else:
            filename_prefix += f"_{n_clusters}"

        # 添加运行标识
        if run_id:
            filename_prefix += f"_{run_id}"

        output_path = os.path.join(output_dir, f"{filename_prefix}_clustering.png")
        visualize_clustering(features_scaled, labels, image_names, visualization_title, output_path)

    return labels, metrics, features_scaled


def run_all_methods(image_dir, output_dir, n_clusters=5, max_images=None, auto_clusters=False, run_id=None):
    """
    运行所有组合（传统特征+层次聚类，深度特征+层次聚类）的聚类方法
    Run all combinations of clustering methods (traditional+hierarchical, deep+hierarchical)

    Args:
        image_dir (str): 图像目录
                         Image directory
        output_dir (str): 输出目录
                          Output directory
        n_clusters (int, optional): 簇的数量（用于层次聚类）
                                    Number of clusters (for hierarchical clustering)
        max_images (int, optional): 每种类型的最大图像数量
                                    Maximum number of images for each type
        auto_clusters (bool, optional): 是否自动确定最佳聚类数量
                                        Whether to automatically determine the optimal number of clusters
        run_id (str, optional): 运行标识，用于唯一文件名
                               Run identifier for unique filenames
    """
    # 创建输出目录
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 生成运行标识（如果未提供）
    # Generate run identifier (if not provided)
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 定义图像类型和特征类型
    # Define image types and feature types
    image_types = ['wavelet', 'markov', 'multichannel']
    feature_types = ['traditional', 'deep']

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
            print(f"\n----- {img_type} + {feature_type} features + hierarchical clustering -----")

            # 设置可视化标题
            # Set visualization title
            visualization_title = f"{img_type.capitalize()} + {feature_type.capitalize()} Features + Hierarchical Clustering"

            # 运行聚类
            # Run clustering
            try:
                _, metrics, _ = run_clustering(
                    image_paths=image_paths,
                    image_names=image_names,
                    feature_type=feature_type,
                    output_dir=img_output_dir,
                    visualization_title=visualization_title,
                    auto_clusters=auto_clusters,
                    n_clusters=n_clusters,
                    image_type=img_type,
                    run_id=run_id
                )

                # 存储结果
                # Store results
                result_key = f"{img_type}_{feature_type}_hierarchical"
                all_results[result_key] = metrics
            except Exception as e:
                print(f"Error processing {img_type}_{feature_type}_hierarchical: {str(e)}")
                continue

    # 生成结果比较表格
    # Generate results comparison table
    generate_results_table(all_results, output_dir, run_id=run_id)


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
    parser.add_argument('--method', type=str, default='traditional_hierarchical',
                        choices=['traditional_hierarchical', 'deep_hierarchical', 'run_all'],
                        help='Clustering method to use')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for hierarchical clustering')
    parser.add_argument('--auto_clusters', action='store_true',
                        help='Automatically determine optimal number of clusters')

    # 输出参数
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='clustering_results',
                        help='Directory to save clustering results')

    args = parser.parse_args()

    # 创建带有参数信息的输出目录
    # Create output directory with parameter information
    output_dir = args.output_dir

    # 添加参数信息到输出目录名称
    # Add parameter information to output directory name
    if args.method != 'run_all':
        # 对于单一方法，添加图像类型和方法信息
        # For single method, add image type and method information
        output_dir = f"{output_dir}_{args.image_type}_{args.method}"
        if args.auto_clusters:
            output_dir += "_auto"
        else:
            output_dir += f"_{args.n_clusters}"
    else:
        # 对于run_all方法，添加时间戳
        # For run_all method, add timestamp
        from datetime import datetime
        output_dir = f"{output_dir}_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.auto_clusters:
            output_dir += "_auto"
        else:
            output_dir += f"_{args.n_clusters}"

    os.makedirs(output_dir, exist_ok=True)

    # 生成运行ID
    # Generate run ID
    from datetime import datetime
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 设置开始时间
    # Set start time
    start_time = time.time()

    # 运行指定的方法
    # Run specified method
    if args.method == 'run_all' or args.image_type == 'all':
        print("\n===== Running all methods =====")
        # 如果选择了所有图像类型，单独处理
        # If all image types are selected, process separately
        if args.image_type == 'all':
            run_all_methods(args.image_dir, output_dir,
                            n_clusters=args.n_clusters,
                            max_images=args.max_images,
                            auto_clusters=args.auto_clusters,
                            run_id=run_id)
        else:
            # 如果只选择了run_all方法但图像类型特定，则单独处理该类型
            # If only run_all method is selected but image type is specific, process that type
            image_paths, image_names = load_images(args.image_dir, args.image_type, args.max_images)
            if len(image_paths) == 0:
                print(f"Error: No images found in {args.image_dir} for type {args.image_type}")
                sys.exit(1)

            results = {}

            # 运行传统特征+层次聚类
            print("\n----- Processing traditional_hierarchical -----")
            visualization_title = f"{args.image_type.capitalize()} + Traditional Features + Hierarchical Clustering"
            try:
                _, metrics_trad, _ = run_clustering(
                    image_paths=image_paths,
                    image_names=image_names,
                    feature_type='traditional',
                    output_dir=output_dir,
                    visualization_title=visualization_title,
                    auto_clusters=args.auto_clusters,
                    n_clusters=args.n_clusters,
                    image_type=args.image_type,
                    run_id=run_id
                )
                results[f"{args.image_type}_traditional_hierarchical"] = metrics_trad
            except Exception as e:
                print(f"Error in traditional hierarchical clustering: {str(e)}")

            # 运行深度特征+层次聚类
            print("\n----- Processing deep_hierarchical -----")
            visualization_title = f"{args.image_type.capitalize()} + Deep Features + Hierarchical Clustering"
            try:
                _, metrics_deep, _ = run_clustering(
                    image_paths=image_paths,
                    image_names=image_names,
                    feature_type='deep',
                    output_dir=output_dir,
                    visualization_title=visualization_title,
                    auto_clusters=args.auto_clusters,
                    n_clusters=args.n_clusters,
                    image_type=args.image_type,
                    run_id=run_id
                )
                results[f"{args.image_type}_deep_hierarchical"] = metrics_deep
            except Exception as e:
                print(f"Error in deep hierarchical clustering: {str(e)}")

            # 生成结果比较表格
            generate_results_table(results, output_dir, run_id=run_id)
    else:
        # 分解方法名称
        # Decompose method name
        feature_type, _ = args.method.split('_')

        # 加载图像
        # Load images
        image_paths, image_names = load_images(args.image_dir, args.image_type, args.max_images)

        if len(image_paths) == 0:
            print(f"Error: No images found in {args.image_dir} for type {args.image_type}")
            sys.exit(1)

        # 设置可视化标题
        # Set visualization title
        visualization_title = f"{args.image_type.capitalize()} + {feature_type.capitalize()} Features + Hierarchical Clustering"

        # 运行聚类 - 修复：添加新的参数
        run_clustering(
            image_paths=image_paths,
            image_names=image_names,
            feature_type=feature_type,
            output_dir=output_dir,
            visualization_title=visualization_title,
            auto_clusters=args.auto_clusters,
            n_clusters=args.n_clusters,
            image_type=args.image_type,
            run_id=run_id
        )

    # 计算运行时间
    # Calculate run time
    end_time = time.time()
    run_time = end_time - start_time

    print(f"\n===== Clustering completed in {run_time:.2f} seconds =====")
    print(f"Results have been saved to the '{output_dir}' directory")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)