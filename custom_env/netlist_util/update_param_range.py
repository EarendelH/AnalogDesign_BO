import yaml
import re
import os
import argparse


def parse_value_with_unit(value_str):
    """Parse a value string that may contain a unit (e.g., '4.0u')"""
    match = re.match(r'([\d.]+)([a-zA-Z]*)', str(value_str))
    if match:
        value, unit = match.groups()
        return float(value), unit
    return float(value_str), ''


def format_value_with_unit(value, unit, decimals=1):
    """Format a value with its unit, keeping one decimal place"""
    return f"{value:.{decimals}f}{unit}"


def calculate_range(base_value):
    """Calculate ±50% range from base value"""
    value, unit = parse_value_with_unit(base_value)
    min_val = value * 0.5  # -50%
    max_val = value * 1.5  # +50%
    return format_value_with_unit(min_val, unit), format_value_with_unit(max_val, unit)


def update_param_ranges(input_dir):
    """Update parameter ranges based on init values in the specified directory"""
    # Construct file paths
    init_param_path = os.path.join(input_dir, 'init_param.yaml')
    param_range_path = os.path.join(input_dir, 'param_range.yaml')
    output_path = os.path.join(input_dir, 'param_range_updated.yaml')

    # Verify input files exist
    if not os.path.exists(init_param_path) or not os.path.exists(param_range_path):
        raise FileNotFoundError(f"Required input files not found in directory: {input_dir}")

    # Read the files
    with open(init_param_path, 'r') as f:
        init_params = yaml.safe_load(f)

    with open(param_range_path, 'r') as f:
        param_ranges = yaml.safe_load(f)

    # Update ranges based on init values
    for component, config in param_ranges.items():
        if component == 'other_variable':
            # Handle special case for other_variable
            for param in config['params']:
                var_name = param['variable_name']
                if var_name in init_params:
                    base_value = init_params[var_name]
                    min_val = int(base_value * 0.5)
                    max_val = int(base_value * 1.5)
                    param['value']['range'] = [min_val, max_val]
            continue

        if 'params' not in config:
            continue

        for param in config['params']:
            var_name = param['variable_name']
            if var_name in init_params:
                base_value = init_params[var_name]
                min_val, max_val = calculate_range(base_value)
                param['value']['range'] = [min_val, max_val]

    # Write updated param_range.yaml
    with open(output_path, 'w') as f:
        yaml.dump(param_ranges, f, default_flow_style=False, sort_keys=False)

    print(f"Updated parameter ranges have been written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Update parameter ranges based on init values')
    parser.add_argument('input_dir', help='Directory containing init_param.yaml and param_range.yaml')
    args = parser.parse_args()

    try:
        update_param_ranges(args.input_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()