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


def check_param_match(init_params, param_ranges):
    init_keys = set(init_params.keys())
    range_keys = set(param_ranges.keys())

    missing_in_init = range_keys - init_keys
    missing_in_range = init_keys - range_keys

    if missing_in_init or missing_in_range:
        print("Parameters do not match between files:")
        if missing_in_init:
            print(f"Missing in init_param: {missing_in_init}")
        if missing_in_range:
            print(f"Missing in param_range: {missing_in_range}")
        return False
    return True


def check_param_compliance(init_params, param_ranges):
    compliant = True
    for param, value in init_params.items():
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
                            print(f"{param}.{sub_param_name}: {init_value} is out of range [{range_min}, {range_max}]")
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
                elif not is_close((init_value - range_min) % step, 0) and not is_close((init_value - range_min) % step,
                                                                                       step):
                    print(f"{param}: {init_value} does not comply with step {step}")
                    compliant = False
    return compliant


def main():
    init_params = load_yaml('init_param.yaml')
    param_ranges = load_yaml('param_range.yaml')

    if check_param_match(init_params, param_ranges):
        print("All parameters match between files.")
        if check_param_compliance(init_params, param_ranges):
            print("All parameters comply with range and step requirements.")
        else:
            print("Some parameters do not comply with range or step requirements.")
    else:
        print("Parameter check aborted due to mismatched parameters.")


if __name__ == "__main__":
    main()