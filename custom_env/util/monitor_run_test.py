import os
import time
import shutil


def clean_folder(target_folder):
    # 获取当前时间
    current_time = time.time()

    # 遍历目标文件夹的第一级子目录
    for entry in os.scandir(target_folder):
        if entry.is_dir():
            dir_path = entry.path
            # 检查目录最后修改时间
            if current_time - os.path.getmtime(dir_path) > 300:  # 300秒为5分钟
                # 遍历第一级子文件夹中的内容
                for sub_entry in os.scandir(dir_path):
                    sub_path = sub_entry.path
                    if sub_entry.is_dir():
                        # 递归删除所有第二级子文件夹
                        shutil.rmtree(sub_path)
                        print(f"Deleted folder: {sub_path}")
                    elif sub_entry.is_file():
                        # 删除所有非 .scs 和 .pkl 的文件
                        if not (sub_entry.name.endswith('.scs') or sub_entry.name.endswith('.pkl')):
                            os.remove(sub_path)
                            print(f"Deleted file: {sub_path}")


def main_loop(target_folder):
    try:
        while True:
            clean_folder(target_folder)
            time.sleep(60)  # 每60秒运行一次
    except KeyboardInterrupt:
        print("Script stopped by user.")


monitor_dir = input("Please input the directory you want to monitor: ")
main_loop(monitor_dir)
