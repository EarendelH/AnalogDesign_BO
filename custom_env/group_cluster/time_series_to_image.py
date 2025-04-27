import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from skimage.feature import graycomatrix
import cv2
from matplotlib.colors import LinearSegmentedColormap

# 导入现有的data_loader模块
from data_loader import load_data, preprocess_signals


def create_output_directories(methods):
    """
    创建输出目录
    Create output directories

    Args:
        methods (list): 需要创建的方法目录列表
                       List of method directories to create
    """
    base_dir = "image_representations"
    os.makedirs(base_dir, exist_ok=True)

    created_dirs = {}
    for method in methods:
        method_dir = os.path.join(base_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        created_dirs[method] = method_dir

    print(f"Created output directories in '{base_dir}'")
    return created_dirs


def normalize_signal(signal_data, min_val=0, max_val=1):
    """
    将信号归一化到指定范围
    Normalize signal to specified range

    Args:
        signal_data (numpy.ndarray): 原始信号数据
                                     Original signal data
        min_val (float, optional): 最小归一化值
                                  Minimum normalization value
        max_val (float, optional): 最大归一化值
                                  Maximum normalization value

    Returns:
        numpy.ndarray: 归一化后的信号
                      Normalized signal
    """
    signal_min = np.min(signal_data)
    signal_max = np.max(signal_data)

    if signal_max == signal_min:
        return np.zeros_like(signal_data) + (min_val + max_val) / 2

    normalized = min_val + (signal_data - signal_min) * (max_val - min_val) / (signal_max - signal_min)
    return normalized


def wavelet_transform_image(signal_data, wavelet='morl', scales=64, figsize=(10, 8)):
    """
    使用连续小波变换生成图像表示
    Generate image representation using continuous wavelet transform

    Args:
        signal_data (numpy.ndarray): 信号数据
                                    Signal data
        wavelet (str, optional): 小波类型
                                Wavelet type
        scales (int, optional): 尺度数量
                               Number of scales
        figsize (tuple, optional): 图像尺寸
                                  Figure size

    Returns:
        numpy.ndarray: 小波变换系数（用于图像表示）
                      Wavelet transform coefficients (for image representation)
    """
    # 标准化信号以便更好地进行小波变换
    # Normalize signal for better wavelet transform
    normalized_signal = normalize_signal(signal_data)

    # 创建尺度列表 - 对数分布
    # Create scales list - logarithmic distribution
    scales = np.logspace(0.1, 1, num=scales)

    # 计算连续小波变换
    # Calculate continuous wavelet transform
    coefficients, frequencies = pywt.cwt(normalized_signal, scales, wavelet)

    # 返回系数的绝对值，可用于可视化和进一步处理
    # Return absolute coefficients for visualization and further processing
    return np.abs(coefficients)


def markov_transition_field(signal_data, n_bins=10, figsize=(10, 8)):
    """
    计算马尔可夫转移场
    Calculate Markov Transition Field

    Args:
        signal_data (numpy.ndarray): 信号数据
                                    Signal data
        n_bins (int, optional): 量化等级数量
                               Number of quantization levels
        figsize (tuple, optional): 图像尺寸
                                  Figure size

    Returns:
        numpy.ndarray: 马尔可夫转移场（用于图像表示）
                      Markov Transition Field (for image representation)
    """
    # 标准化信号
    # Normalize signal
    normalized_signal = normalize_signal(signal_data)

    # 信号量化为n_bins个等级
    # Quantize signal into n_bins levels
    bins = np.linspace(0, 1, n_bins + 1)
    quantized = np.digitize(normalized_signal, bins) - 1
    quantized = np.clip(quantized, 0, n_bins - 1)  # 确保在有效范围内

    # 计算一步转移概率矩阵
    # Calculate one-step transition probability matrix
    n = len(quantized)
    Q = np.zeros((n_bins, n_bins))

    for i in range(n - 1):
        Q[quantized[i], quantized[i + 1]] += 1

    # 行归一化以获得概率
    # Row normalize to get probabilities
    row_sums = Q.sum(axis=1)
    # 避免除以零
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    Q = Q / row_sums[:, np.newaxis]

    # 创建马尔可夫转移场
    # Create Markov Transition Field
    MTF = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            MTF[i, j] = Q[quantized[i], quantized[j]]

    return MTF


def multi_channel_image(signal_data, figsize=(10, 8)):
    """
    创建多通道图像表示：原始信号、一阶导数和二阶导数
    Create multi-channel image representation: original signal, first derivative, and second derivative

    Args:
        signal_data (numpy.ndarray): 信号数据
                                    Signal data
        figsize (tuple, optional): 图像尺寸
                                  Figure size

    Returns:
        numpy.ndarray: 三通道图像数组（用于图像表示）
                      Three-channel image array (for image representation)
    """
    # 标准化原始信号（用于红色通道）
    # Normalize original signal (for red channel)
    channel_1 = normalize_signal(signal_data)

    # 计算一阶导数并标准化（用于绿色通道）
    # Calculate first derivative and normalize (for green channel)
    first_derivative = np.gradient(signal_data)
    channel_2 = normalize_signal(first_derivative)

    # 计算二阶导数并标准化（用于蓝色通道）
    # Calculate second derivative and normalize (for blue channel)
    second_derivative = np.gradient(first_derivative)
    channel_3 = normalize_signal(second_derivative)

    # 创建三通道图像 shape: (height, width, channels)
    # Create three-channel image
    # 为了展示成2D图像，我们将每个通道转换为一个n×n的图像，其中n是信号长度
    # To visualize as 2D image, we'll transform each channel into an n×n image where n is signal length
    n = len(signal_data)

    # 方法1：创建彩色图像，每个通道代表信号的不同方面
    # Method 1: Create a color image with each channel representing different aspects of the signal
    img_3d = np.zeros((n, n, 3))

    # 填充三个通道
    # Fill the three channels
    for i in range(n):
        img_3d[:, i, 0] = channel_1  # 红色通道 - 原始信号
        img_3d[:, i, 1] = channel_2  # 绿色通道 - 一阶导数
        img_3d[:, i, 2] = channel_3  # 蓝色通道 - 二阶导数

    return img_3d


def save_image(img_data, output_path, title, colormap='viridis', vmin=None, vmax=None):
    """
    保存图像数据到文件
    Save image data to file

    Args:
        img_data (numpy.ndarray): 图像数据
                                 Image data
        output_path (str): 输出文件路径
                          Output file path
        title (str): 图像标题
                    Image title
        colormap (str, optional): 颜色映射
                                 Colormap
        vmin (float, optional): 颜色映射最小值
                               Minimum value for colormap
        vmax (float, optional): 颜色映射最大值
                               Maximum value for colormap
    """
    plt.figure(figsize=(10, 8))

    # 检查是否为彩色图像（多通道）
    # Check if it's a color image (multi-channel)
    if img_data.ndim == 3 and img_data.shape[2] == 3:
        plt.imshow(img_data)
    else:
        plt.imshow(img_data, cmap=colormap, aspect='auto', vmin=vmin, vmax=vmax)

    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Image saved to {output_path}")


def process_wavelet_transform(signals, signal_names, output_dir):
    """
    处理所有信号的小波变换并保存图像
    Process wavelet transform for all signals and save images

    Args:
        signals (list): 信号数据列表
                       List of signal data
        signal_names (list): 信号名称列表
                           List of signal names
        output_dir (str): 输出目录
                         Output directory
    """
    print("\nProcessing Wavelet Transform images...")

    for i, (signal, name) in enumerate(zip(signals, signal_names)):
        # 计算小波变换
        # Calculate wavelet transform
        cwt_image = wavelet_transform_image(signal)

        # 保存图像
        # Save image
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = os.path.join(output_dir, f"wavelet_{safe_name}.png")
        save_image(cwt_image, output_path, f"Wavelet Transform: {name}",
                   colormap='jet', vmin=0)

        # 显示进度
        # Show progress
        if (i + 1) % 10 == 0 or (i + 1) == len(signals):
            print(f"  Processed {i + 1}/{len(signals)} signals")


def process_markov_field(signals, signal_names, output_dir, n_bins=10):
    """
    处理所有信号的马尔可夫转移场并保存图像
    Process Markov Transition Field for all signals and save images

    Args:
        signals (list): 信号数据列表
                       List of signal data
        signal_names (list): 信号名称列表
                           List of signal names
        output_dir (str): 输出目录
                         Output directory
        n_bins (int, optional): 量化等级数量
                               Number of quantization levels
    """
    print("\nProcessing Markov Transition Field images...")

    for i, (signal, name) in enumerate(zip(signals, signal_names)):
        # 计算马尔可夫转移场
        # Calculate Markov Transition Field
        mtf_image = markov_transition_field(signal, n_bins=n_bins)

        # 保存图像
        # Save image
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = os.path.join(output_dir, f"markov_{safe_name}.png")
        save_image(mtf_image, output_path, f"Markov Transition Field: {name}",
                   colormap='plasma')

        # 显示进度
        # Show progress
        if (i + 1) % 10 == 0 or (i + 1) == len(signals):
            print(f"  Processed {i + 1}/{len(signals)} signals")


def process_multi_channel(signals, signal_names, output_dir):
    """
    处理所有信号的多通道图像表示并保存图像
    Process multi-channel image representation for all signals and save images

    Args:
        signals (list): 信号数据列表
                       List of signal data
        signal_names (list): 信号名称列表
                           List of signal names
        output_dir (str): 输出目录
                         Output directory
    """
    print("\nProcessing Multi-Channel images...")

    for i, (signal, name) in enumerate(zip(signals, signal_names)):
        # 创建多通道图像
        # Create multi-channel image
        multi_channel = multi_channel_image(signal)

        # 保存图像
        # Save image
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = os.path.join(output_dir, f"multichannel_{safe_name}.png")
        save_image(multi_channel, output_path, f"Multi-Channel Image: {name}")

        # 显示进度
        # Show progress
        if (i + 1) % 10 == 0 or (i + 1) == len(signals):
            print(f"  Processed {i + 1}/{len(signals)} signals")


def main():
    """
    主函数
    Main function
    """
    # 解析命令行参数
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Time Series to Image Converter')

    # 数据输入参数
    # Data input parameters
    parser.add_argument('--file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--handle_missing', type=str, default='report',
                        choices=['report', 'drop', 'fill_mean', 'fill_zero'],
                        help='Strategy for handling missing values')

    # 转换方法选择
    # Conversion method selection
    parser.add_argument('--method', type=str, default='all',
                        choices=['wavelet', 'markov', 'multichannel', 'all'],
                        help='Image conversion method to use')

    # 马尔可夫转移场参数
    # Markov Transition Field parameters
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of quantization bins for Markov Transition Field')

    # 小波变换参数
    # Wavelet transform parameters
    parser.add_argument('--wavelet', type=str, default='morl',
                        help='Wavelet type for continuous wavelet transform')
    parser.add_argument('--scales', type=int, default=64,
                        help='Number of scales for wavelet transform')

    args = parser.parse_args()

    # 确定要使用的方法
    # Determine methods to use
    methods = []
    if args.method == 'all':
        methods = ['wavelet', 'markov', 'multichannel']
    else:
        methods = [args.method]

    # 创建输出目录
    # Create output directories
    output_dirs = create_output_directories(methods)

    # 加载数据
    # Load data
    print(f"\n===== Loading time series data from {args.file} =====")
    time_points, signal_names, signals = load_data(args.file, handle_missing=args.handle_missing)
    print(f"Loaded {len(signals)} signals with {len(time_points)} time points each")

    # 预处理信号（Z-score标准化）
    # Preprocess signals (Z-score normalization)
    preprocessed_signals = preprocess_signals(signals)
    print("Signals preprocessed with Z-score normalization")

    # 应用选定的转换方法
    # Apply selected conversion methods
    if 'wavelet' in methods:
        process_wavelet_transform(preprocessed_signals, signal_names, output_dirs['wavelet'])

    if 'markov' in methods:
        process_markov_field(preprocessed_signals, signal_names, output_dirs['markov'],
                             n_bins=args.n_bins)

    if 'multichannel' in methods:
        process_multi_channel(preprocessed_signals, signal_names, output_dirs['multichannel'])

    print("\n===== Time series to image conversion completed successfully =====")
    print("Results have been saved to the 'image_representations' directory")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)