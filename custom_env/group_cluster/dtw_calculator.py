import numpy as np
from tslearn.metrics import cdist_dtw
import sys


def calculate_dtw_distance_matrix(signals, sakoe_chiba_radius=None):
    """
    Calculate DTW distance matrix for all pairs of signals
    计算所有信号对之间的DTW距离矩阵

    Args:
        signals (list): List of signal arrays
                        信号数组列表
        sakoe_chiba_radius (int, optional): Sakoe-Chiba band radius (None means no constraint)
                                            Sakoe-Chiba带半径（None表示无约束）

    Returns:
        numpy.ndarray: DTW distance matrix
                      DTW距离矩阵
    """
    try:
        # Convert list of signals to 3D array required by tslearn
        # 将信号列表转换为tslearn所需的3D数组
        signals_array = np.array([s.reshape(-1, 1) for s in signals])

        # Calculate the DTW distance matrix
        # 计算DTW距离矩阵
        print("Calculating DTW distance matrix...")
        dtw_matrix = cdist_dtw(signals_array, signals_array,
                               global_constraint="sakoe_chiba" if sakoe_chiba_radius else None,
                               sakoe_chiba_radius=sakoe_chiba_radius)

        print(f"DTW distance matrix calculated with shape: {dtw_matrix.shape}")
        return dtw_matrix

    except MemoryError:
        print("Error: Out of memory when calculating DTW distance matrix.")
        print("Try to use a constraint (sakoe_chiba_radius) to reduce memory usage.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in DTW calculation: {str(e)}")
        sys.exit(1)