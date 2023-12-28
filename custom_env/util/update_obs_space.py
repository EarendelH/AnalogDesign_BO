import gymnasium.spaces as spaces
import re
from collections import OrderedDict
import math

import numpy as np

from util.util_func import unit_conversion


def update_obs_space(ideal_specs, cur_specs, cur_param):
    """
    Update observation space for the environment.
    :param ideal_specs: Dict, ideal normalized specs for the circuit
    :param cur_specs: Dict, current normalized specs for the circuit
    :param cur_param: OrderedDict, current device parameters deleting unit of the circuit
    :return: gym.spaces.Dict, the observation space for the environment
    """

    print("Debug, in update_obs_space, ideal_specs = ", ideal_specs)
    print("Debug, in update_obs_space, cur_specs = ", cur_specs)
    print("Debug, in update_obs_space, cur_param = ", cur_param)

    # Convert cur_param to Dict, and convert unit to float
    cur_param_dict = {}
    for key, value in cur_param.items():
        cur_param_dict[key] = np.array([unit_conversion(value)], dtype=np.float32)

    # Flatten ideal_specs and cur_specs
    ideal_specs_flatten = {key: specs['value'] for key, specs in ideal_specs.items()}
    cur_specs_flatten = {k: v for d in cur_specs.values() for k, v in d.items()}

    # Convert ideal_specs_flatten and cur_specs_flatten values to np.array
    for key, value in ideal_specs_flatten.items():
        print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {value}")
        ideal_specs_flatten[key] = np.array([value], dtype=np.float32)
        print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {ideal_specs_flatten[key]}")
    for key, value in cur_specs_flatten.items():
        print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {value}")
        cur_specs_flatten[key] = np.array([value], dtype=np.float32)
        print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {cur_specs_flatten[key]}")

    # Combine three dicts into one
    dict_sum = {"cur_specs": cur_specs_flatten, "ideal_specs": ideal_specs_flatten, "cur_param": cur_param_dict}
    dict_sum = OrderedDict(dict_sum)

    return dict_sum

def update_obs_space_simple(ideal_specs, cur_specs, cur_param):
    """
    Update observation space for the environment.
    :param ideal_specs: Dict, ideal normalized specs for the circuit
    :param cur_specs: Dict, current normalized specs for the circuit
    :param cur_param: OrderedDict, current device parameters deleting unit of the circuit
    :return: gym.spaces.Dict, the observation space for the environment
    """

    print("Debug, in update_obs_space, ideal_specs = ", ideal_specs)
    print("Debug, in update_obs_space, cur_specs = ", cur_specs)
    print("Debug, in update_obs_space, cur_param = ", cur_param)

    # Convert cur_param to Dict, and convert unit to float
    cur_param_dict = {}
    for key, value in cur_param.items():
        cur_param_dict[key] = np.array([unit_conversion(value)], dtype=np.float32)

    # Flatten ideal_specs and cur_specs
    ideal_specs_flatten = {key: specs['value'] for key, specs in ideal_specs.items()}
    cur_specs_flatten = {k: v for d in cur_specs.values() for k, v in d.items()}

    # Convert ideal_specs_flatten and cur_specs_flatten values to np.array
    for key, value in ideal_specs_flatten.items():
        print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {value}")
        ideal_specs_flatten[key] = np.array([value], dtype=np.float32)
        print(f"Debug, in update_obs_space, ideal_specs_flatten[{key}] = {ideal_specs_flatten[key]}")
    for key, value in cur_specs_flatten.items():
        print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {value}")
        cur_specs_flatten[key] = np.array([value], dtype=np.float32)
        print(f"Debug, in update_obs_space, cur_specs_flatten[{key}] = {cur_specs_flatten[key]}")

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
# update_obs_space(ideal_specs, cur_specs, cur_param) print(obs_space)

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
