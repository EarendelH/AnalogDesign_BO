import yaml
import re
import math

EPSILON = 1e-9


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def parse_value(value):
    if isinstance(value, (int, float)):
        return value
    match = re.match(r'(\d+(?:\.\d+)?)(p|n|u|m|k|M|G)?', str(value))
    if match:
        num, unit = match.groups()
        num = float(num)
        if unit == 'p':
            return num * 1e-12
        elif unit == 'n':
            return num * 1e-9
        elif unit == 'u':
            return num * 1e-6
        elif unit == 'm':
            return num * 1e-3
        elif unit == 'k':
            return num * 1e3
        elif unit == 'M':
            return num * 1e6
        elif unit == 'G':
            return num * 1e9
        else:
            return num
    return value


def is_close(a, b, rel_tol=EPSILON, abs_tol=EPSILON):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def extract_param_sets(init_params):
    param_sets = []
    current_set = {}
    for item in init_params:
        if isinstance(item, dict):
            if current_set:
                param_sets.append(current_set)
                current_set = {}
            current_set.update(item)
        elif isinstance(item, str):
            key, value = item.split(': ')
            current_set[key] = value
    if current_set:
        param_sets.append(current_set)
    return param_sets


def check_param_match(init_params, param_ranges):
    param_sets = extract_param_sets(init_params)
    all_init_keys = set()
    for param_set in param_sets:
        all_init_keys.update(param_set.keys())
    range_keys = set(param_ranges.keys())

    missing_in_init = range_keys - all_init_keys
    missing_in_range = all_init_keys - range_keys

    if missing_in_init or missing_in_range:
        print("Parameters do not match between files:")
        if missing_in_init:
            print(f"Missing in init_param: {missing_in_init}")
        if missing_in_range:
            print(f"Missing in param_range: {missing_in_range}")
        return False
    return True


def check_param_compliance(init_params, param_ranges):
    param_sets = extract_param_sets(init_params)
    all_compliant = True
    for i, param_set in enumerate(param_sets):
        print(f"\nChecking parameter set {i + 1}:")
        compliant = True
        for param, value in param_set.items():
            if param in param_ranges:
                if isinstance(param_ranges[param], dict) and 'params' in param_ranges[param]:
                    for sub_param in param_ranges[param]['params']:
                        sub_param_name = sub_param['variable_name']
                        if sub_param_name in value:
                            init_value = parse_value(value[sub_param_name])
                            range_min = parse_value(sub_param['value']['range'][0])
                            range_max = parse_value(sub_param['value']['range'][1])
                            step = parse_value(sub_param['value']['step'])

                            if not (range_min <= init_value <= range_max):
                                print(
                                    f"{param}.{sub_param_name}: {init_value} is out of range [{range_min}, {range_max}]")
                                compliant = False
                            elif not is_close((init_value - range_min) % step, 0) and not is_close(
                                    (init_value - range_min) % step, step):
                                print(f"{param}.{sub_param_name}: {init_value} does not comply with step {step}")
                                compliant = False
                elif isinstance(param_ranges[param], dict) and 'value' in param_ranges[param]:
                    init_value = parse_value(value)
                    range_min = parse_value(param_ranges[param]['value']['range'][0])
                    range_max = parse_value(param_ranges[param]['value']['range'][1])
                    step = parse_value(param_ranges[param]['value']['step'])

                    if not (range_min <= init_value <= range_max):
                        print(f"{param}: {init_value} is out of range [{range_min}, {range_max}]")
                        compliant = False
                    elif not is_close((init_value - range_min) % step, 0) and not is_close(
                            (init_value - range_min) % step, step):
                        print(f"{param}: {init_value} does not comply with step {step}")
                        compliant = False
        if compliant:
            print(f"All parameters in set {i + 1} comply with range and step requirements.")
        else:
            print(f"Some parameters in set {i + 1} do not comply with range or step requirements.")
        all_compliant = all_compliant and compliant
    return all_compliant


def main():
    init_params = load_yaml('init_param.yaml')
    param_ranges = load_yaml('param_range.yaml')

    if check_param_match(init_params, param_ranges):
        print("All parameters match between files.")
        if check_param_compliance(init_params, param_ranges):
            print("\nAll parameter sets comply with range and step requirements.")
        else:
            print("\nSome parameter sets do not comply with range or step requirements.")
    else:
        print("Parameter check aborted due to mismatched parameters.")


if __name__ == "__main__":
    main()