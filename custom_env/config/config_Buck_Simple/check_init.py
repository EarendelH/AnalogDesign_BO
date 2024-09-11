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


def extract_param_ranges(param_ranges):
    extracted_ranges = {}
    for key, value in param_ranges.items():
        if key == 'other_variable':
            for param in value['params']:
                extracted_ranges[param['variable_name']] = param['value']
        elif 'params' in value:
            for param in value['params']:
                extracted_ranges[f"{key}.{param['variable_name']}"] = param['value']
    return extracted_ranges


def check_param_compliance(init_params, param_ranges):
    param_sets = extract_param_sets(init_params)
    extracted_ranges = extract_param_ranges(param_ranges)
    all_compliant = True

    for i, param_set in enumerate(param_sets):
        print(f"\nChecking parameter set {i + 1}:")
        set_compliant = True
        for param, value in param_set.items():
            range_key = param
            if '.' not in param:
                # Check if this is a device parameter
                for device in param_ranges:
                    if device != 'other_variable' and param in [p['variable_name'] for p in
                                                                param_ranges[device]['params']]:
                        range_key = f"{device}.{param}"
                        break

            if range_key in extracted_ranges:
                init_value = parse_value(value)
                range_min = parse_value(extracted_ranges[range_key]['range'][0])
                range_max = parse_value(extracted_ranges[range_key]['range'][1])
                step = parse_value(extracted_ranges[range_key]['step'])

                if not (range_min <= init_value <= range_max):
                    print(f"{param}: {init_value} is out of range [{range_min}, {range_max}]")
                    set_compliant = False
                elif not is_close((init_value - range_min) % step, 0) and not is_close((init_value - range_min) % step,
                                                                                       step):
                    print(f"{param}: {init_value} does not comply with step {step}")
                    set_compliant = False
            else:
                print(f"Warning: {param} not found in param_range.yaml")

        if set_compliant:
            print(f"All parameters in set {i + 1} comply with range and step requirements.")
        else:
            print(f"Some parameters in set {i + 1} do not comply with range or step requirements.")
        all_compliant = all_compliant and set_compliant

    return all_compliant


def main():
    init_params = load_yaml('init_param.yaml')
    param_ranges = load_yaml('param_range.yaml')

    if check_param_compliance(init_params, param_ranges):
        print("\nAll parameter sets comply with range and step requirements.")
    else:
        print("\nSome parameter sets do not comply with range or step requirements.")


if __name__ == "__main__":
    main()
