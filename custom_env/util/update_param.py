from collections import OrderedDict
import yaml
from numpy import array
from typing import Dict, Tuple, Union


# Helper function: Split a value into its numeric and unit parts.
def split_value_unit(value: Union[str, int]) -> Tuple[float, str]:
    if isinstance(value, int):
        return float(value), ''  # Convert int to float and return with an empty unit
    else:
        for i, char in enumerate(value):
            if not char.isdigit() and char != '.':
                return float(value[:i]), value[i:]  # Split the string into numeric and unit parts
        return float(value), ''


# Helper function: Combine a numeric value with a unit into a string.
def combine_value_unit(value: float, unit: str) -> str:
    return f"{value}{unit}"


# Helper function: Adjust a value within a given range.
def adjust_value_within_range(value: str, step: str, range_limits: list, action: int, is_integer: bool = False) -> str:
    value_num, unit = split_value_unit(value)
    step_num, _ = split_value_unit(step)
    min_limit, max_limit = split_value_unit(range_limits[0])[0], split_value_unit(range_limits[1])[0]

    if action == 0:  # Decrease
        value_num = max(min_limit, value_num - step_num)
    elif action == 2:  # Increase
        value_num = min(max_limit, value_num + step_num)
    elif action == 1:  # Keep the same
        pass

    if is_integer:  # Convert to integer if needed
        value_num = int(round(value_num))

    return combine_value_unit(value_num, unit)


# Main function: Update parameters based on the action_idx_dict and cur_param_dict
def update_parameters(action_idx_dict, cur_param_dict, yaml_config):
    updated_params = OrderedDict()

    with open(yaml_config, 'r') as file:
        yaml_config = yaml.safe_load(file)

    for key, actions in action_idx_dict.items():
        if key.startswith('M'):
            for i, action in enumerate(actions):
                param_keys = [f"w_{key}_per_finger", f"l_{key}", f"nf_{key}"]  # Construct parameter keys
                if i < len(param_keys):
                    param_key = param_keys[i]
                    current_value = cur_param_dict[param_key]
                    step_info = yaml_config[key]['params'][i]['value']['step']
                    range_info = yaml_config[key]['params'][i]['value']['range']
                    is_integer = param_key.startswith("nf_")  # Check if the parameter is of type nf_X
                    updated_value = adjust_value_within_range(current_value, step_info, range_info, action, is_integer)
                    updated_params[param_key] = updated_value
        else:
            print(f"Debug: {cur_param_dict[key]}")
            current_value = cur_param_dict[key]
            step_info = yaml_config['other_variable']['params'][0]['value']['step']
            range_info = yaml_config['other_variable']['params'][0]['value']['range']
            updated_value = adjust_value_within_range(current_value, step_info, range_info, actions[0])
            updated_params[key] = updated_value

    return updated_params


# Test Code
# action_idx = OrderedDict( [('M13', array([2, 1, 2])), ('M14', array([2, 2, 1])), ('M16', array([2, 0,
# 1])), ('M23', array([0, 1, 0])), ('M24', array([2, 2, 0])), ('M25', array([0, 0, 2])), ('M35', array([2, 0, 0])),
# ('M36', array([2, 0, 2])), ('M17', array([0, 1, 2])), ('M18', array([1, 0, 2])), ('M11', array([1, 2, 2])), ('M12',
# array([1, 0, 1])), ('M19', array([1, 0, 2])), ('M20', array([2, 0, 0])), ('M21', array([0, 1, 2])), ('M22',
# array([1, 1, 1])), ('IB', array([1]))])
# cur_param_dict = OrderedDict( [('w_M14_per_finger', '5.0u'), ('l_M14',
# '5.0u'), ('nf_M14', '10'), ('w_M35_per_finger', '5.0u'), ('l_M35', '5.0u'), ('nf_M35', '10'), ('w_M25_per_finger',
# '5.0u'), ('l_M25', '5.0u'), ('nf_M25', '10'), ('w_M13_per_finger', '5.0u'), ('l_M13', '5.0u'), ('nf_M13', '10'),
# ('w_M12_per_finger', '5.0u'), ('l_M12', '5.0u'), ('nf_M12', '10'), ('w_M11_per_finger', '5.0u'), ('l_M11', '5.0u'),
# ('nf_M11', '10'), ('w_M20_per_finger', '5.0u'), ('l_M20', '5.0u'), ('nf_M20', '10'), ('w_M19_per_finger', '5.0u'),
# ('l_M19', '5.0u'), ('nf_M19', '10'), ('w_M36_per_finger', '5.0u'), ('l_M36', '5.0u'), ('nf_M36', '10'),
# ('w_M16_per_finger', '5.0u'), ('l_M16', '5.0u'), ('nf_M16', '10'), ('w_M24_per_finger', '5.0u'), ('l_M24', '5.0u'),
# ('nf_M24', '10'), ('w_M23_per_finger', '5.0u'), ('l_M23', '5.0u'), ('nf_M23', '10'), ('w_M22_per_finger', '5.0u'),
# ('l_M22', '5.0u'), ('nf_M22', '10'), ('w_M21_per_finger', '5.0u'), ('l_M21', '5.0u'), ('nf_M21', '10'),
# ('w_M18_per_finger', '5.0u'), ('l_M18', '5.0u'), ('nf_M18', '10'), ('w_M17_per_finger', '5.0u'), ('l_M17', '5.0u'),
# ('nf_M17', '10'), ('IB', '25.0u')])
# yaml_config = yaml.safe_load(open("../config_3/param_range.yaml",
# 'r'))
# updated_params = update_parameters(action_idx, cur_param_dict, yaml_config)
# print(updated_params)

# Output output = OrderedDict([('w_M13_per_finger', "5.5u"), ('l_M13', '5.0u'), ('nf_M13', '11'),
# ('w_M14_per_finger', '5.5u'), ('l_M14', '5.5u'), ('nf_M14', '10'), ('w_M16_per_finger', '5.5u'), ('l_M16', '4.5u'),
# ('nf_M16', '10'), ('w_M23_per_finger', '4.5u'), ('l_M23', '5.0u'), ('nf_M23', '9'), ('w_M24_per_finger', '5.5u'),
# ('l_M24', '5.5u'), ('nf_M24', '9'), ('w_M25_per_finger', '4.5u'), ('l_M25', '4.5u'), ('nf_M25', '11'),
# ('w_M35_per_finger', '5.5u'), ('l_M35', '4.5u'), ('nf_M35', '9'), ('w_M36_per_finger', '5.5u'), ('l_M36', '4.5u'),
# ('nf_M36', '11'), ('w_M17_per_finger', '4.5u'), ('l_M17', '5.0u'), ('nf_M17', '11'), ('w_M18_per_finger', '5.0u'),
# ('l_M18', '4.5u'), ('nf_M18', '11'), ('w_M11_per_finger', '5.0u'), ('l_M11', '5.5u'), ('nf_M11', '11'),
# ('w_M12_per_finger', '5.0u'), ('l_M12', '4.5u'), ('nf_M12', '10'), ('w_M19_per_finger', '5.0u'), ('l_M19', '4.5u'),
# ('nf_M19', '11'), ('w_M20_per_finger', '5.5u'), ('l_M20', '4.5u'), ('nf_M20', '9'), ('w_M21_per_finger', '4.5u'),
# ('l_M21', '5.0u'), ('nf_M21', '11'), ('w_M22_per_finger', '5.0u'), ('l_M22', '5.0u'), ('nf_M22', '10'), ('IB',
# '25.0u')])
