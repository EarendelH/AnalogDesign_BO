import os
import shutil
import re


def process_scs_files(directory_path):
    # 定义所有可能的参数组合
    corner1_values = ['ff', 'ss', 'fs', 'sf']
    corner2_values = ['ff', 'ss']
    corner3_values = ['ff', 'ss']
    temp_values = ['b40', '125']
    esr_values = ['p2', '1']  # 新增 ESR 值
    vddi_values = ['1p65', '1p95']  # 新增 VDDI 值

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('_tt.scs'):
            base_filename = filename[:-7]  # 移除 '_tt.scs'

            # 为每个组合创建新文件
            for corner1 in corner1_values:
                for corner2 in corner2_values:
                    for corner3 in corner3_values:
                        for temp in temp_values:
                            for esr in esr_values:
                                for vddi in vddi_values:
                                    # 构建新文件名
                                    new_filename = f"{base_filename}_{corner1}_{corner2}_{corner3}_{temp}_{esr}_{vddi}.scs"
                                    source_path = os.path.join(directory_path, filename)
                                    dest_path = os.path.join(directory_path, new_filename)

                                    # 复制文件
                                    shutil.copy2(source_path, dest_path)

                                    # 读取文件内容
                                    with open(dest_path, 'r', encoding='utf-8') as file:
                                        content = file.read()

                                    # 进行文本替换
                                    # 1. 替换第一个corner (TT -> FF/SF/FS/SS)
                                    content = re.sub(
                                        r'(section=)TT',
                                        f'\\1{corner1.upper()}',
                                        content
                                    )

                                    # 2. 替换第二个corner (RES_ALL_TT -> RES_ALL_FF/RES_ALL_SS)
                                    content = re.sub(
                                        r'(section=)RES_ALL_TT',
                                        f'\\1RES_ALL_{corner2.upper()}',
                                        content
                                    )

                                    # 3. 替换第三个corner (MOM_ALL_TT -> MOM_ALL_FF/MOM_ALL_SS)
                                    content = re.sub(
                                        r'(section=)MOM_ALL_TT',
                                        f'\\1MOM_ALL_{corner3.upper()}',
                                        content
                                    )

                                    # 4. 替换温度值
                                    temp_value = '-40.0' if temp == 'b40' else '125.0'
                                    content = re.sub(
                                        r'temp=27.0',
                                        f'temp={temp_value}',
                                        content
                                    )

                                    # 5. 替换 ESR 值
                                    esr_value = '0.2' if esr == 'p2' else '1'
                                    content = re.sub(
                                        r'ESR=0.2',
                                        f'ESR={esr_value}',
                                        content
                                    )

                                    # 6. 替换 VDDI 值
                                    vddi_value = '1.65' if vddi == '1p65' else '1.95'
                                    content = re.sub(
                                        r'VDDI=1.65',
                                        f'VDDI={vddi_value}',
                                        content
                                    )

                                    # 写回文件
                                    with open(dest_path, 'w', encoding='utf-8') as file:
                                        file.write(content)

                                    print(f"Created and processed: {new_filename}")


def main():
    # 获取用户输入的目录路径
    directory_path = input("请输入包含SCS文件的目录路径: ")

    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print("错误：指定的目录不存在！")
        return

    try:
        process_scs_files(directory_path)
        print("处理完成！")
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")


if __name__ == "__main__":
    main()