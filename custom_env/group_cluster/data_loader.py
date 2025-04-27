import pandas as pd
import numpy as np
import sys


def load_data(file_path):
    """
    Load time series data from CSV file
    加载CSV文件中的时间序列数据

    Args:
        file_path (str): Path to the CSV file
                         CSV文件路径

    Returns:
        tuple: (time_points, signal_names, signals)
               时间点、信号名称、信号数据
    """
    try:
        # Read CSV file
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # Check if dataframe is empty
        # 检查数据框是否为空
        if df.empty:
            print("Error: The CSV file is empty.")
            sys.exit(1)

        # Extract time points (first column)
        # 提取时间点（第一列）
        time_column_name = df.columns[0]
        time_points = df[time_column_name].values

        # Extract signal names and signals
        # 提取信号名称和信号数据
        signal_names = df.columns[1:].tolist()
        signals = [df[name].values for name in signal_names]

        # Check if all signals have the same length
        # 检查所有信号是否具有相同的长度
        signal_lengths = [len(signal) for signal in signals]
        if len(set(signal_lengths)) > 1:
            print("Error: Not all signals have the same length.")
            sys.exit(1)

        # Check for missing values
        # 检查是否有缺失值
        if df.isnull().any().any():
            print("Error: Missing values detected in the data.")
            sys.exit(1)

        return time_points, signal_names, signals

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def preprocess_signals(signals):
    """
    Preprocess signals (Z-score normalization)
    预处理信号（Z分数标准化）

    Args:
        signals (list): List of signal arrays
                        信号数组列表

    Returns:
        list: List of preprocessed signal arrays
              预处理后的信号数组列表
    """
    preprocessed_signals = []

    for signal in signals:
        # Check if signal has variance (to avoid division by zero)
        # 检查信号是否有方差（避免除以零）
        if np.std(signal) == 0:
            print("Error: Signal with zero variance detected.")
            sys.exit(1)

        # Z-score normalization
        # Z分数标准化
        normalized_signal = (signal - np.mean(signal)) / np.std(signal)
        preprocessed_signals.append(normalized_signal)

    return preprocessed_signals