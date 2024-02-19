import yaml
from collections import OrderedDict
import numpy as np

from util.util_func import unit_conversion
# from util_func import unit_conversion


def update_obs_space(ideal_specs_dict, cur_specs_dict, cur_param_input):
    """
    Update observation space for the environment.
    :param ideal_specs_dict: Dict, ideal normalized specs for the circuit
    :param cur_specs_dict: Dict, current normalized specs for the circuit
    :param cur_param_input: OrderedDict, current device parameters deleting unit of the circuit
    :return: gym.spaces.Dict, the observation space for the environment
    """

    # print("Debug, in update_obs_space, ideal_specs = ", ideal_specs)
    # print("Debug, in update_obs_space, cur_specs = ", cur_specs)
    # print("Debug, in update_obs_space, cur_param = ", cur_param)

    # Convert cur_param to Dict, and convert unit to float
    cur_param_dict = {}
    for key, value in cur_param_input.items():
        cur_param_dict[key] = np.array([unit_conversion(value)], dtype=np.float32)

    # Flatten ideal_specs and cur_specs
    ideal_specs_flatten = {k: v for d in ideal_specs_dict.values() for k, v in d.items()}
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}

    # Convert ideal_specs_flatten and cur_specs_flatten values to np.array
    for key, value in ideal_specs_flatten.items():
        # print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {value}")
        ideal_specs_flatten[key] = np.array([value], dtype=np.float32)
        # print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {ideal_specs_flatten[key]}")
    for key, value in cur_specs_flatten.items():
        # print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {value}")
        cur_specs_flatten[key] = np.array([value], dtype=np.float32)
        # print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {cur_specs_flatten[key]}")

    # Combine three dicts into one
    dict_sum = {"cur_specs": cur_specs_flatten, "ideal_specs": ideal_specs_flatten, "cur_param": cur_param_dict}
    dict_sum = OrderedDict(dict_sum)

    return dict_sum


def update_obs_space_simple(ideal_specs_dict, cur_specs_dict, cur_param_input):
    """
    Update observation space for the environment.
    :param ideal_specs_dict: Dict, ideal normalized specs for the circuit
    :param cur_specs_dict: Dict, current normalized specs for the circuit
    :param cur_param_input: OrderedDict, current device parameters deleting unit of the circuit
    :return: gym.spaces.Dict, the observation space for the environment
    """

    # print("Debug, in update_obs_space, ideal_specs = ", ideal_specs)
    # print("Debug, in update_obs_space, cur_specs = ", cur_specs)
    # print("Debug, in update_obs_space, cur_param = ", cur_param)

    # Convert cur_param to Dict, and convert unit to float
    cur_param_dict = {}
    for key, value in cur_param_input.items():
        cur_param_dict[key] = np.array([unit_conversion(value)], dtype=np.float32)

    # Flatten ideal_specs and cur_specs
    ideal_specs_flatten = {key: specs['value'] for key, specs in ideal_specs_dict.items()}
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}

    # Convert ideal_specs_flatten and cur_specs_flatten values to np.array
    for key, value in ideal_specs_flatten.items():
        # print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {value}")
        ideal_specs_flatten[key] = np.array([value], dtype=np.float32)
        # print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {ideal_specs_flatten[key]}")
    for key, value in cur_specs_flatten.items():
        # print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {value}")
        cur_specs_flatten[key] = np.array([value], dtype=np.float32)
        # print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {cur_specs_flatten[key]}")

    # Combine three dicts into one
    dict_sum = {"cur_specs": cur_specs_flatten}
    dict_sum = OrderedDict(dict_sum)

    return dict_sum


# Test Code
# cur_specs = {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': 0, 'gainBandWidth': 0},
# 'Trans': {'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081}, 'PSRR': {'powerSupplyRejectionRatio':
# 9.24983891912481}}
# ideal_specs = {'DC': {'pwr': 0.1}, 'Stability': {'phaseMargin': 1, 'gainBandWidth': 1},
# 'Trans': {'slewRateUp': 500.5, 'slewRateDown': 500.5}, 'PSRR': {'powerSupplyRejectionRatio': 10}}
# cur_param =
# OrderedDict( [('w_M13_per_finger', '4.5u'), ('l_M13', '4.5u'), ('nf_M13', '20'), ('w_M14_per_finger', '1.0u'),
# ('l_M14', '1.5u'), ('nf_M14', '3'), ('w_M16_per_finger', '1.0u'), ('l_M16', '1.5u'), ('nf_M16', '3'),
# ('w_M23_per_finger', '0.5u'), ('l_M23', '0.5u'), ('nf_M23', '1'), ('w_M24_per_finger', '1.0u'), ('l_M24', '1.0u'),
# ('nf_M24', '2'), ('w_M25_per_finger', '1.5u'), ('l_M25', '1.5u'), ('nf_M25', '2'), ('w_M35_per_finger', '1.5u'),
# ('l_M35', '1.0u'), ('nf_M35', '3'), ('w_M36_per_finger', '1.5u'), ('l_M36', '0.5u'), ('nf_M36', '3'),
# ('w_M17_per_finger', '1.5u'), ('l_M17', '1.0u'), ('nf_M17', '2'), ('w_M18_per_finger', '1.0u'), ('l_M18', '1.5u'),
# ('nf_M18', '2'), ('w_M11_per_finger', '0.5u'), ('l_M11', '0.5u'), ('nf_M11', '2'), ('w_M12_per_finger', '1.5u'),
# ('l_M12', '1.0u'), ('nf_M12', '1'), ('w_M19_per_finger', '1.5u'), ('l_M19', '1.0u'), ('nf_M19', '3'),
# ('w_M20_per_finger', '1.0u'), ('l_M20', '1.0u'), ('nf_M20', '1'), ('w_M21_per_finger', '1.0u'), ('l_M21', '1.0u'),
# ('nf_M21', '1'), ('w_M22_per_finger', '0.5u'), ('l_M22', '1.5u'), ('nf_M22', '2'), ('IB', '50.0u')])
# obs_space =
# update_obs_space(ideal_specs, cur_specs, cur_param)
# print(obs_space)

# Output OrderedDict([('cur_specs', {'pwr': array([0.0008036], dtype=float32), 'phaseMargin': array([0.],
# dtype=float32), 'gainBandWidth': array([0.], dtype=float32), 'slewRateUp': array([4996857.5], dtype=float32),
# 'slewRateDown': array([5684722.], dtype=float32), 'powerSupplyRejectionRatio': array([9.249839], dtype=float32)}),
# ('ideal_specs', {'pwr': array([0.1], dtype=float32), 'phaseMargin': array([1.], dtype=float32), 'gainBandWidth':
# array([1.], dtype=float32), 'slewRateUp': array([500.5], dtype=float32), 'slewRateDown': array([500.5],
# dtype=float32), 'powerSupplyRejectionRatio': array([10.], dtype=float32)}), ('cur_param', {'w_M13_per_finger':
# array([4.5e-06], dtype=float32), 'l_M13': array([4.5e-06], dtype=float32), 'nf_M13': array([20.], dtype=float32),
# 'w_M14_per_finger': array([1.e-06], dtype=float32), 'l_M14': array([1.5e-06], dtype=float32), 'nf_M14': array([3.],
# dtype=float32), 'w_M16_per_finger': array([1.e-06], dtype=float32), 'l_M16': array([1.5e-06], dtype=float32),
# 'nf_M16': array([3.], dtype=float32), 'w_M23_per_finger': array([5.e-07], dtype=float32), 'l_M23': array([5.e-07],
# dtype=float32), 'nf_M23': array([1.], dtype=float32), 'w_M24_per_finger': array([1.e-06], dtype=float32),
# 'l_M24': array([1.e-06], dtype=float32), 'nf_M24': array([2.], dtype=float32), 'w_M25_per_finger': array([1.5e-06],
# dtype=float32), 'l_M25': array([1.5e-06], dtype=float32), 'nf_M25': array([2.], dtype=float32), 'w_M35_per_finger':
# array([1.5e-06], dtype=float32), 'l_M35': array([1.e-06], dtype=float32), 'nf_M35': array([3.], dtype=float32),
# 'w_M36_per_finger': array([1.5e-06], dtype=float32), 'l_M36': array([5.e-07], dtype=float32), 'nf_M36': array([3.],
# dtype=float32), 'w_M17_per_finger': array([1.5e-06], dtype=float32), 'l_M17': array([1.e-06], dtype=float32),
# 'nf_M17': array([2.], dtype=float32), 'w_M18_per_finger': array([1.e-06], dtype=float32), 'l_M18': array([1.5e-06],
# dtype=float32), 'nf_M18': array([2.], dtype=float32), 'w_M11_per_finger': array([5.e-07], dtype=float32),
# 'l_M11': array([5.e-07], dtype=float32), 'nf_M11': array([2.], dtype=float32), 'w_M12_per_finger': array([1.5e-06],
# dtype=float32), 'l_M12': array([1.e-06], dtype=float32), 'nf_M12': array([1.], dtype=float32), 'w_M19_per_finger':
# array([1.5e-06], dtype=float32), 'l_M19': array([1.e-06], dtype=float32), 'nf_M19': array([3.], dtype=float32),
# 'w_M20_per_finger': array([1.e-06], dtype=float32), 'l_M20': array([1.e-06], dtype=float32), 'nf_M20': array([1.],
# dtype=float32), 'w_M21_per_finger': array([1.e-06], dtype=float32), 'l_M21': array([1.e-06], dtype=float32),
# 'nf_M21': array([1.], dtype=float32), 'w_M22_per_finger': array([5.e-07], dtype=float32), 'l_M22': array([1.5e-06],
# dtype=float32), 'nf_M22': array([2.], dtype=float32), 'IB': array([5.e-05], dtype=float32)})])


def flatten_observation(observation):
    # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
    sorted_cur_specs = OrderedDict(sorted(observation['cur_specs'].items()))
    sorted_ideal_specs = OrderedDict(sorted(observation['ideal_specs'].items()))
    sorted_cur_param = OrderedDict(sorted(observation['cur_param'].items()))

    # Print the sorted dicts for debugging
    # print("Sorted cur_specs:", sorted_cur_specs)
    # print("Sorted ideal_specs:", sorted_ideal_specs)
    # print("Sorted cur_param:", sorted_cur_param)

    # Combine all values from sorted dicts into a single tuple
    combined_values = tuple(
        list(sorted_cur_specs.values()) +
        list(sorted_ideal_specs.values()) +
        list(sorted_cur_param.values())
    )

    return combined_values


# Test Code
# cur_specs = {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': 0, 'gainBandWidth': 0},
#              'Trans': {'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081},
#              'PSRR': {'powerSupplyRejectionRatio': 9.24983891912481}}
# ideal_specs = {'DC': {'pwr': 0.1}, 'Stability': {'phaseMargin': 1, 'gainBandWidth': 1},
#                'Trans': {'slewRateUp': 500.5, 'slewRateDown': 500.5}, 'PSRR': {'powerSupplyRejectionRatio': 10}}
# cur_param = OrderedDict(
#     [('w_M13_per_finger', '4.5u'), ('l_M13', '4.5u'), ('nf_M13', '20'), ('w_M14_per_finger', '1.0u'),
#      ('l_M14', '1.5u'), ('nf_M14', '3'), ('w_M16_per_finger', '1.0u'), ('l_M16', '1.5u'), ('nf_M16', '3'),
#      ('w_M23_per_finger', '0.5u'), ('l_M23', '0.5u'), ('nf_M23', '1'), ('w_M24_per_finger', '1.0u'),
#      ('l_M24', '1.0u'),
#      ('nf_M24', '2'), ('w_M25_per_finger', '1.5u'), ('l_M25', '1.5u'), ('nf_M25', '2'), ('w_M35_per_finger', '1.5u'),
#      ('l_M35', '1.0u'), ('nf_M35', '3'), ('w_M36_per_finger', '1.5u'), ('l_M36', '0.5u'), ('nf_M36', '3'),
#      ('w_M17_per_finger', '1.5u'), ('l_M17', '1.0u'), ('nf_M17', '2'), ('w_M18_per_finger', '1.0u'),
#      ('l_M18', '1.5u'),
#      ('nf_M18', '2'), ('w_M11_per_finger', '0.5u'), ('l_M11', '0.5u'), ('nf_M11', '2'), ('w_M12_per_finger', '1.5u'),
#      ('l_M12', '1.0u'), ('nf_M12', '1'), ('w_M19_per_finger', '1.5u'), ('l_M19', '1.0u'), ('nf_M19', '3'),
#      ('w_M20_per_finger', '1.0u'), ('l_M20', '1.0u'), ('nf_M20', '1'), ('w_M21_per_finger', '1.0u'),
#      ('l_M21', '1.0u'),
#      ('nf_M21', '1'), ('w_M22_per_finger', '0.5u'), ('l_M22', '1.5u'), ('nf_M22', '2'), ('IB', '50.0u')])
# obs_space = update_obs_space(ideal_specs, cur_specs, cur_param)
# print(obs_space)
# print(flatten_observation(obs_space))

def update_obs_space_w_type(ideal_specs_dict, cur_specs_dict, cur_param_input, param_range_config_path):
    """
    Update observation space for the environment.
    :param param_range_config_path: str, the path of the param_range_config_file
    :param ideal_specs_dict: Dict, ideal normalized specs for the circuit
    :param cur_specs_dict: Dict, current normalized specs for the circuit
    :param cur_param_input: OrderedDict, current device parameters deleting unit of the circuit
    :return: gym.spaces.Dict, the observation space for the environment
    """

    # print("Debug, in update_obs_space, ideal_specs = ", ideal_specs)
    # print("Debug, in update_obs_space, cur_specs = ", cur_specs)
    # print("Debug, in update_obs_space, cur_param = ", cur_param)

    with open(param_range_config_path, 'r') as file:
        param_range_config = yaml.safe_load(file)

    # Convert cur_param to Dict, and convert unit to float
    cur_param_dict = {}
    for key, value in cur_param_input.items():
        cur_param_dict[key] = np.array([unit_conversion(value)], dtype=np.float32)

    # Flatten ideal_specs and cur_specs
    ideal_specs_flatten = {k: v for d in ideal_specs_dict.values() for k, v in d.items()}
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}

    # Convert ideal_specs_flatten and cur_specs_flatten values to np.array
    for key, value in ideal_specs_flatten.items():
        # print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {value}")
        ideal_specs_flatten[key] = np.array([value], dtype=np.float32)
        # print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {ideal_specs_flatten[key]}")
    for key, value in cur_specs_flatten.items():
        # print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {value}")
        cur_specs_flatten[key] = np.array([value], dtype=np.float32)
        # print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {cur_specs_flatten[key]}")

    device_type = {}
    # Add type sub-dict
    # 0 for pch, 1 for nch, 2 for vsource, 3 for isource
    for key, value in param_range_config.items():
        if key == 'other_variable':
            for param in value['params']:
                device_name = param['variable_name']
                if device_name.startswith('I'):
                    device_type[device_name] = 3
                elif device_name.startswith('V'):
                    device_type[device_name] = 2
        else:
            if value['instance_type'] == 'nch_mac':
                device_type[key] = 1
            if value['instance_type'] == 'pch_mac':
                device_type[key] = 0

    # Combine three dicts into one
    dict_sum = {"cur_specs": cur_specs_flatten, "ideal_specs": ideal_specs_flatten, "cur_param": cur_param_dict,
                "device_type": device_type}
    dict_sum = OrderedDict(dict_sum)

    return dict_sum


def flatten_observation_w_type(observation):
    # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
    sorted_cur_specs = OrderedDict(sorted(observation['cur_specs'].items()))
    sorted_ideal_specs = OrderedDict(sorted(observation['ideal_specs'].items()))
    sorted_cur_param = OrderedDict(sorted(observation['cur_param'].items()))
    sorted_device_type = OrderedDict(sorted(observation['device_type'].items()))

    # Print the sorted dicts for debugging
    # print("Sorted cur_specs:", sorted_cur_specs)
    # print("Sorted ideal_specs:", sorted_ideal_specs)
    # print("Sorted cur_param:", sorted_cur_param)

    # Combine all values from sorted dicts into a single tuple
    combined_values = tuple(
        list(sorted_cur_specs.values()) +
        list(sorted_ideal_specs.values()) +
        list(sorted_cur_param.values()) +
        list(sorted_device_type.values())
    )

    return combined_values


# Test Code
# cur_specs = {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': 0, 'gainBandWidth': 0},
#              'Trans': {'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081},
#              'PSRR': {'powerSupplyRejectionRatio': 9.24983891912481}}
# ideal_specs = {'DC': {'pwr': 0.1}, 'Stability': {'phaseMargin': 1, 'gainBandWidth': 1},
#                'Trans': {'slewRateUp': 500.5, 'slewRateDown': 500.5}, 'PSRR': {'powerSupplyRejectionRatio': 10}}
# cur_param = OrderedDict(
#     [('w_M13_per_finger', '4.5u'), ('l_M13', '4.5u'), ('nf_M13', '20'), ('w_M14_per_finger', '1.0u'),
#      ('l_M14', '1.5u'), ('nf_M14', '3'), ('w_M16_per_finger', '1.0u'), ('l_M16', '1.5u'), ('nf_M16', '3'),
#      ('w_M23_per_finger', '0.5u'), ('l_M23', '0.5u'), ('nf_M23', '1'), ('w_M24_per_finger', '1.0u'),
#      ('l_M24', '1.0u'),
#      ('nf_M24', '2'), ('w_M25_per_finger', '1.5u'), ('l_M25', '1.5u'), ('nf_M25', '2'), ('w_M35_per_finger', '1.5u'),
#      ('l_M35', '1.0u'), ('nf_M35', '3'), ('w_M36_per_finger', '1.5u'), ('l_M36', '0.5u'), ('nf_M36', '3'),
#      ('w_M17_per_finger', '1.5u'), ('l_M17', '1.0u'), ('nf_M17', '2'), ('w_M18_per_finger', '1.0u'),
#      ('l_M18', '1.5u'),
#      ('nf_M18', '2'), ('w_M11_per_finger', '0.5u'), ('l_M11', '0.5u'), ('nf_M11', '2'), ('w_M12_per_finger', '1.5u'),
#      ('l_M12', '1.0u'), ('nf_M12', '1'), ('w_M19_per_finger', '1.5u'), ('l_M19', '1.0u'), ('nf_M19', '3'),
#      ('w_M20_per_finger', '1.0u'), ('l_M20', '1.0u'), ('nf_M20', '1'), ('w_M21_per_finger', '1.0u'),
#      ('l_M21', '1.0u'),
#      ('nf_M21', '1'), ('w_M22_per_finger', '0.5u'), ('l_M22', '1.5u'), ('nf_M22', '2'), ('IB', '50.0u')])
# param_range_config_file = '../config_3/param_range.yaml'
# obs_space = update_obs_space_w_type(ideal_specs, cur_specs, cur_param, param_range_config_file)
# print(obs_space)
# print(flatten_observation_w_type(obs_space))

# Output

# OrderedDict([('cur_specs', {'pwr': array([0.0008036], dtype=float32), 'phaseMargin': array([0.],
# dtype=float32), 'gainBandWidth': array([0.], dtype=float32), 'slewRateUp': array([4996857.5], dtype=float32),
# 'slewRateDown': array([5684722.], dtype=float32), 'powerSupplyRejectionRatio': array([9.249839], dtype=float32)}),
# ('ideal_specs', {'pwr': array([0.1], dtype=float32), 'phaseMargin': array([1.], dtype=float32), 'gainBandWidth':
# array([1.], dtype=float32), 'slewRateUp': array([500.5], dtype=float32), 'slewRateDown': array([500.5],
# dtype=float32), 'powerSupplyRejectionRatio': array([10.], dtype=float32)}), ('cur_param', {'w_M13_per_finger':
# array([4.5e-06], dtype=float32), 'l_M13': array([4.5e-06], dtype=float32), 'nf_M13': array([20.], dtype=float32),
# 'w_M14_per_finger': array([1.e-06], dtype=float32), 'l_M14': array([1.5e-06], dtype=float32), 'nf_M14': array([3.],
# dtype=float32), 'w_M16_per_finger': array([1.e-06], dtype=float32), 'l_M16': array([1.5e-06], dtype=float32),
# 'nf_M16': array([3.], dtype=float32), 'w_M23_per_finger': array([5.e-07], dtype=float32), 'l_M23': array([5.e-07],
# dtype=float32), 'nf_M23': array([1.], dtype=float32), 'w_M24_per_finger': array([1.e-06], dtype=float32),
# 'l_M24': array([1.e-06], dtype=float32), 'nf_M24': array([2.], dtype=float32), 'w_M25_per_finger': array([1.5e-06],
# dtype=float32), 'l_M25': array([1.5e-06], dtype=float32), 'nf_M25': array([2.], dtype=float32), 'w_M35_per_finger':
# array([1.5e-06], dtype=float32), 'l_M35': array([1.e-06], dtype=float32), 'nf_M35': array([3.], dtype=float32),
# 'w_M36_per_finger': array([1.5e-06], dtype=float32), 'l_M36': array([5.e-07], dtype=float32), 'nf_M36': array([3.],
# dtype=float32), 'w_M17_per_finger': array([1.5e-06], dtype=float32), 'l_M17': array([1.e-06], dtype=float32),
# 'nf_M17': array([2.], dtype=float32), 'w_M18_per_finger': array([1.e-06], dtype=float32), 'l_M18': array([1.5e-06],
# dtype=float32), 'nf_M18': array([2.], dtype=float32), 'w_M11_per_finger': array([5.e-07], dtype=float32),
# 'l_M11': array([5.e-07], dtype=float32), 'nf_M11': array([2.], dtype=float32), 'w_M12_per_finger': array([1.5e-06],
# dtype=float32), 'l_M12': array([1.e-06], dtype=float32), 'nf_M12': array([1.], dtype=float32), 'w_M19_per_finger':
# array([1.5e-06], dtype=float32), 'l_M19': array([1.e-06], dtype=float32), 'nf_M19': array([3.], dtype=float32),
# 'w_M20_per_finger': array([1.e-06], dtype=float32), 'l_M20': array([1.e-06], dtype=float32), 'nf_M20': array([1.],
# dtype=float32), 'w_M21_per_finger': array([1.e-06], dtype=float32), 'l_M21': array([1.e-06], dtype=float32),
# 'nf_M21': array([1.], dtype=float32), 'w_M22_per_finger': array([5.e-07], dtype=float32), 'l_M22': array([1.5e-06],
# dtype=float32), 'nf_M22': array([2.], dtype=float32), 'IB': array([5.e-05], dtype=float32)}), ('device_type',
# {'M14': 2, 'M35': 2, 'M25': 2, 'M13': 2, 'M12': 2, 'M11': 2, 'M20': 1, 'M19': 1, 'M36': 1, 'M16': 1, 'M24': 1,
# 'M23': 1, 'M22': 1, 'M21': 1, 'M18': 1, 'M17': 1, 'IB': 4})])
#
# (array([0.], dtype=float32), array([0.],
# dtype=float32), array([9.249839], dtype=float32), array([0.0008036], dtype=float32), array([5684722.],
# dtype=float32), array([4996857.5], dtype=float32), array([1.], dtype=float32), array([1.], dtype=float32),
# array([10.], dtype=float32), array([0.1], dtype=float32), array([500.5], dtype=float32), array([500.5],
# dtype=float32), array([5.e-05], dtype=float32), array([5.e-07], dtype=float32), array([1.e-06], dtype=float32),
# array([4.5e-06], dtype=float32), array([1.5e-06], dtype=float32), array([1.5e-06], dtype=float32), array([1.e-06],
# dtype=float32), array([1.5e-06], dtype=float32), array([1.e-06], dtype=float32), array([1.e-06], dtype=float32),
# array([1.e-06], dtype=float32), array([1.5e-06], dtype=float32), array([5.e-07], dtype=float32), array([1.e-06],
# dtype=float32), array([1.5e-06], dtype=float32), array([1.e-06], dtype=float32), array([5.e-07], dtype=float32),
# array([2.], dtype=float32), array([1.], dtype=float32), array([20.], dtype=float32), array([3.], dtype=float32),
# array([3.], dtype=float32), array([2.], dtype=float32), array([2.], dtype=float32), array([3.], dtype=float32),
# array([1.], dtype=float32), array([1.], dtype=float32), array([2.], dtype=float32), array([1.], dtype=float32),
# array([2.], dtype=float32), array([2.], dtype=float32), array([3.], dtype=float32), array([3.], dtype=float32),
# array([5.e-07], dtype=float32), array([1.5e-06], dtype=float32), array([4.5e-06], dtype=float32), array([1.e-06],
# dtype=float32), array([1.e-06], dtype=float32), array([1.5e-06], dtype=float32), array([1.e-06], dtype=float32),
# array([1.5e-06], dtype=float32), array([1.e-06], dtype=float32), array([1.e-06], dtype=float32), array([5.e-07],
# dtype=float32), array([5.e-07], dtype=float32), array([1.e-06], dtype=float32), array([1.5e-06], dtype=float32),
# array([1.5e-06], dtype=float32), array([1.5e-06], dtype=float32), 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1)
