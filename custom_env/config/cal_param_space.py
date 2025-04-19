import yaml
import math
import re


def parse_value(value_str):
    """解析带有单位的数值，如 1.0u, 15k, 60n 等"""
    if isinstance(value_str, (int, float)):
        return float(value_str)

    # 匹配数字和单位
    match = re.match(r'([\d.]+)([a-zA-Z]*)', str(value_str))
    if not match:
        raise ValueError(f"无法解析的值: {value_str}")

    number, unit = match.groups()
    number = float(number)

    # 根据单位转换为基本单位
    unit_multipliers = {
        'f': 1e-15,  # femto
        'p': 1e-12,  # pico
        'n': 1e-9,  # nano
        'u': 1e-6,  # micro
        'm': 1e-3,  # milli
        'k': 1e3,  # kilo
        'M': 1e6  # mega
    }

    multiplier = unit_multipliers.get(unit, 1)
    return number * multiplier


def calculate_steps(min_val, max_val, step_val):
    """计算从min到max，以step为步长，有多少个值"""
    min_parsed = parse_value(min_val)
    max_parsed = parse_value(max_val)
    step_parsed = parse_value(step_val)

    # 避免浮点精度问题，加一个小的epsilon
    steps = int((max_parsed - min_parsed) / step_parsed + 1.00001)
    return max(1, steps)  # 至少有一个值


def load_yaml_and_calculate(file_path):
    """加载YAML文件并计算参数空间大小"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # 计算每个设备的参数组合数
    device_combinations = {}

    for device_name, device_info in data.items():
        if 'params' not in device_info:
            continue

        param_count = 1
        param_details = []

        for param in device_info['params']:
            var_name = param['variable_name']
            value_info = param['value']

            if 'range' in value_info and 'step' in value_info:
                min_val, max_val = value_info['range']
                step_val = value_info['step']

                steps = calculate_steps(min_val, max_val, step_val)
                param_count *= steps

                param_details.append({
                    'name': var_name,
                    'range': value_info['range'],
                    'step': value_info['step'],
                    'combinations': steps
                })

        device_combinations[device_name] = {
            'params': param_details,
            'combinations': param_count,
            'log10_combinations': math.log10(param_count) if param_count > 0 else 0
        }

    # 计算总组合数（用对数）
    total_log_combinations = 0
    for device, info in device_combinations.items():
        total_log_combinations += math.log10(info['combinations'])

    return device_combinations, total_log_combinations


def main():
    file_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang_Regroup/param_range.yaml"
    device_combinations, total_log_combinations = load_yaml_and_calculate(file_path)

    # 打印每个设备的组合数
    print("每个设备的参数组合数：")
    print("=" * 60)
    for device, info in device_combinations.items():
        print(f"设备: {device}")
        print(f"组合数: {info['combinations']:,} (log10: {info['log10_combinations']:.2f})")

        print("参数细节:")
        for param in info['params']:
            print(f"  - {param['name']}: {param['range']} (步长: {param['step']}) -> {param['combinations']:,}种可能值")
        print("-" * 60)

    # 打印总组合数（以10为底的对数形式）
    print(f"\n总参数空间大小:")
    print(f"对数形式 (log10): {total_log_combinations:.2f}")
    print(f"科学计数法: 10^{total_log_combinations:.2f} ≈ {10 ** total_log_combinations:.2e}")


if __name__ == "__main__":
    main()