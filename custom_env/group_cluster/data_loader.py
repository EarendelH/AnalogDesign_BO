import pandas as pd
import numpy as np
import sys


def load_data(file_path, handle_missing='report'):
    """
    Load time series data from CSV file
    加载CSV文件中的时间序列数据

    Args:
        file_path (str): Path to the CSV file
                         CSV文件路径
        handle_missing (str): How to handle missing values: 'report', 'drop', 'fill_mean', 'fill_zero'
                             处理缺失值的方式：'report'（报告）, 'drop'（删除）, 'fill_mean'（均值填充）, 'fill_zero'（零填充）

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

        # Check for missing values and handle them according to specified strategy
        # 检查缺失值并根据指定策略处理
        if df.isnull().any().any():
            # Find where missing values are
            # 找出缺失值的位置
            missing_columns = df.columns[df.isnull().any()].tolist()
            missing_rows = []

            for col in missing_columns:
                rows = df[df[col].isnull()].index.tolist()
                for row in rows:
                    missing_rows.append((row, col))

            print(f"Warning: Missing values detected in {len(missing_columns)} column(s).")
            print(f"Missing data locations (row_index, column_name):")
            for row, col in missing_rows:
                print(f"  - Row {row}, Column '{col}'")

            # Handle missing values based on specified strategy
            # 根据指定策略处理缺失值
            if handle_missing == 'report':
                print("Error: Missing values found. Please fix the data or specify a handling strategy.")
                sys.exit(1)

            elif handle_missing == 'drop':
                print("Dropping rows with missing values...")
                df = df.dropna()
                print(f"Remaining rows after dropping: {len(df)}")

                # Update time points after dropping
                time_points = df[time_column_name].values

            elif handle_missing == 'fill_mean':
                print("Filling missing values with column means...")
                df = df.fillna(df.mean())

            elif handle_missing == 'fill_zero':
                print("Filling missing values with zeros...")
                df = df.fillna(0)

            else:
                print(f"Unknown missing value handling strategy: {handle_missing}")
                print("Error: Missing values found. Please fix the data or specify a valid handling strategy.")
                sys.exit(1)

        # Extract signals after handling missing values
        # 处理缺失值后提取信号
        signals = [df[name].values for name in signal_names]

        # Check if all signals have the same length
        # 检查所有信号是否具有相同的长度
        signal_lengths = [len(signal) for signal in signals]
        if len(set(signal_lengths)) > 1:
            print("Error: Not all signals have the same length.")
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