import os
from joblib import load, dump


def merge_and_sort_joblibs(folder_path):
    # Step 2: 定义变量存储所有加载的数据
    all_data = []

    # Step 3: 遍历指定文件夹中的所有joblib文件
    for file in os.listdir(folder_path):
        if file.endswith('.joblib'):
            file_path = os.path.join(folder_path, file)
            data = load(file_path)  # 加载joblib文件
            all_data.append(data)  # 将加载的数据添加到列表中

    # Step 4: 根据'rew'值对列表进行排序
    sorted_data = sorted(all_data, key=lambda x: x['rew'], reverse=True)

    return sorted_data


# 使用函数的示例
folder_path = input("Please enter the path to the folder containing the joblib files: ")
sorted_data = merge_and_sort_joblibs(folder_path)

# 可选: 将排序后的数据保存为一个新的joblib文件
output_path = input("Please enter the name of the new joblib file to save the sorted data: ")
dump(sorted_data, output_path)