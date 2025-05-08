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

        # Check for missing values
        # 检查缺失值
        if df.isnull().any().any():
            # Find where missing values are
            # 找出缺失值的位置
            missing_columns = df.columns[df.isnull().any()].tolist()
            missing_rows = []

            for col in missing_columns:
                rows = df[df[col].isnull()].index.tolist()
                for row in rows:
                    missing_rows.append((row, col))

            print(f"Error: Missing values detected in {len(missing_columns)} column(s).")
            print(f"Missing data locations (row_index, column_name):")
            for row, col in missing_rows:
                print(f"  - Row {row}, Column '{col}'")
            print("Please fix the data before proceeding.")
            sys.exit(1)

        # Extract signals
        # 提取信号
        signals = [df[name].values for name in signal_names]

        # Check if all signals have the same length - modified to warning
        # 检查所有信号是否具有相同的长度 - 修改为警告
        signal_lengths = [len(signal) for signal in signals]
        if len(set(signal_lengths)) > 1:
            print("Warning: Not all signals have the same length.")
            print("Signal lengths:")
            for i, (name, length) in enumerate(zip(signal_names, signal_lengths)):
                print(f"  - Signal {i}: '{name}', Length: {length}")
            print("Signals will be rescaled to the shortest length.")

        return time_points, signal_names, signals

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def rescale_signals_to_min_length(signals, time_points=None):
    """
    Rescale all signals to the length of the shortest signal
    将所有信号缩放到最短信号的长度

    Args:
        signals (list): List of signal arrays
                        信号数组列表
        time_points (numpy.ndarray, optional): Time points array
                                              时间点数组

    Returns:
        tuple: (rescaled_signals, rescaled_time_points)
               缩放后的信号、缩放后的时间点
    """
    # Find the length of the shortest signal
    # 找到最短信号的长度
    signal_lengths = [len(signal) for signal in signals]
    min_length = min(signal_lengths)

    # If all signals already have the same length, return as is
    # 如果所有信号已经具有相同的长度，直接返回
    if len(set(signal_lengths)) == 1:
        if time_points is not None:
            return signals, time_points
        return signals

    print(f"Rescaling all signals to the minimum length: {min_length}")

    # Rescale each signal to the minimum length using interpolation
    # 使用插值将每个信号缩放到最小长度
    rescaled_signals = []
    import scipy.interpolate as interp

    for i, signal in enumerate(signals):
        if len(signal) == min_length:
            # No need to rescale if already at min_length
            # 如果已经是最小长度，则不需要缩放
            rescaled_signals.append(signal)
        else:
            # Create interpolation function
            # 创建插值函数
            x_original = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, min_length)
            f = interp.interp1d(x_original, signal)

            # Apply interpolation
            # 应用插值
            rescaled_signal = f(x_new)
            rescaled_signals.append(rescaled_signal)

            # Print rescaling info
            # 打印缩放信息
            print(f"  - Signal {i} rescaled from length {len(signal)} to {min_length}")

    # Also rescale time_points if provided
    # 如果提供了时间点，也对其进行缩放
    rescaled_time_points = None
    if time_points is not None:
        if len(time_points) > min_length:
            x_original = np.linspace(0, 1, len(time_points))
            x_new = np.linspace(0, 1, min_length)
            f = interp.interp1d(x_original, time_points)
            rescaled_time_points = f(x_new)
            print(f"  - Time points rescaled from length {len(time_points)} to {min_length}")
        else:
            rescaled_time_points = time_points

        return rescaled_signals, rescaled_time_points

    return rescaled_signals

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