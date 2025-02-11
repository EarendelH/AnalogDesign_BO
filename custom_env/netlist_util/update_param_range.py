import yaml
import re
import os

def parse_value(value_str):
    """解析带单位的数值，返回数值部分和单位"""
    match = re.match(r"([\d.]+)(.*)", str(value_str))
    if match:
        number = float(match.group(1))
        unit = match.group(2)
        return number, unit
    return None, ""

def generate_new_range(init_value):
    """根据初始值生成±50%的范围，保留单位和一位小数"""
    num, unit = parse_value(init_value)
    lower = round(num * 0.5, 1)
    upper = round(num * 1.5, 1)
    return [f"{lower:.1f}{unit}", f"{upper:.1f}{unit}"]

def main():
    # 读取init_param.yaml中的初始值
    with open('init_param.yaml') as f:
        init_params = yaml.safe_load(f)[0]  # 假设是列表第一个元素

    # 读取param_range.yaml
    with open('param_range.yaml') as f:
        param_range = yaml.safe_load(f)

    # 遍历所有组件和参数
    for component in param_range.values():
        for param in component.get('params', []):
            var_name = param['variable_name']
            if var_name in init_params:
                # 生成新范围
                new_range = generate_new_range(init_params[var_name])
                # 保留原始数据类型（列表中的字符串表示）
                param['value']['range'] = [str(x) for x in new_range]

    # 创建输出文件夹
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # 保存修改后的param_range.yaml
    output_path = os.path.join(output_folder, "param_range_updated.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(param_range, f, sort_keys=False, default_flow_style=None)

    # 将init_param.yaml也复制到输出文件夹
    init_param_output_path = os.path.join(output_folder, "init_param.yaml")
    with open(init_param_output_path, 'w') as f:
        yaml.dump([init_params], f, sort_keys=False, default_flow_style=None)

    print(f"处理完成，结果已保存到 {output_folder} 文件夹下的 param_range_updated.yaml 和 init_param.yaml")

if __name__ == "__main__":
    main()