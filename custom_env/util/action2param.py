from collections import OrderedDict
from numpy import array
import yaml
import copy


def action2param(device_mask_flag, device_mask_dict, action_space, param_rang_dict):
    """
    Generate param with given continuous action.
    :param device_mask_flag:
    :param device_mask_dict:
    :param action_space:
    :param param_rang_dict:
    :return: param_dict
    """

    # Flatten action space
    flatten_action_space = {}
    for _, value in action_space.items():
        flatten_action_space.update(value)

    # print(flatten_action_space)

    # Extend action space via mask config
    masked_action_space = copy.deepcopy(flatten_action_space)

    if device_mask_flag:
        for master_device in flatten_action_space:
            if master_device in device_mask_dict:
                slave_devices = device_mask_dict[master_device]
                for slave_device in slave_devices:
                    param_value = flatten_action_space[master_device]
                    masked_action_space[slave_device] = param_value

    # print(masked_action_space)

    extend_action_space = {}
    for key, action in masked_action_space.items():
        if key.startswith('M'):
            for index, value in enumerate(action):
                key_name = None
                if index == 0:
                    key_name = f"w_{key}_per_finger"
                if index == 1:
                    key_name = f"l_{key}"
                if index == 2:
                    key_name = f"nf_{key}"
                extend_action_space[key_name] = value
        else:
            extend_action_space[key] = float(action[0])
    # print(extend_action_space)

    param_dict = OrderedDict()
    for key, value in param_rang_dict.items():
        for param in value['params']:
            variable_name = param['variable_name']
            value_range = param['value']['range']
            step = param['value']['step']
            action_value = extend_action_space[variable_name]
            # Check if the variable name starts with 'nf_' to determine if it should be integer
            is_integer = variable_name.startswith('nf_')
            param_value = parse_action2param(value_range, step, action_value, is_integer)
            # print(variable_name, value_range, step, action_value, is_integer)
            # print(param_value)
            param_dict[variable_name] = param_value

    return param_dict


def parse_action2param(value_range, step, action_value, is_intege):
    """
    Parse action to param.
    :param value_range:
    :param step:
    :param action_value:
    :param is_intege:
    :return: param_value
    """
    min_value, max_value = value_range

    if isinstance(min_value, str):
        magnitude = ''.join(filter(str.isalpha, min_value))
        min_value, max_value, step = [float(x.replace(magnitude, '')) for x in [min_value, max_value, step]]
    else:
        magnitude = ''
        min_value, max_value, step = float(min_value), float(max_value), float(step)

    original_value = min_value + (max_value - min_value) * action_value
    nearest_value = min_value + round((original_value - min_value) / step) * step

    # Check within the range
    if nearest_value < min_value:
        nearest_value = min_value
    if nearest_value > max_value:
        nearest_value = max_value

    if is_intege:
        nearest_value = str(int(nearest_value))
    else:
        nearest_value = f"{nearest_value}{magnitude}"

    return nearest_value


# Test Code

# device_mask_flag = True
# device_mask = "../config/device_mask.yaml"
# with open(device_mask, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# action_space = OrderedDict([('Agent_1', OrderedDict(
#     [('IBP', array([0.58232024])), ('M1', array([0.55986021, 0.71259075, 0.32580201])),
#      ('M4', array([0.66283334, 0.9521516, 0.37748751])), ('M5', array([0.40918823, 0.07680309, 0.44814611]))])),
#                             ('Agent_2', OrderedDict([('M9', array([0.82469542, 0.35882003, 0.36313247]))])),
#                             ('Agent_3', OrderedDict([('M12', array([0.37167306, 0.25930124, 0.37495792]))])), (
#                             'Agent_4', OrderedDict(
#                                 [('CM', array([0.54670056])), ('M0', array([0.54749156, 0.13308131, 0.40636367])),
#                                  ('R0', array([0.61302462])), ('R1', array([0.37650811]))]))])
# param_range = "../config/param_range.yaml"
# with open(param_range, 'r') as file:
#     param_rang_dict = yaml.safe_load(file)
# print(action2param(device_mask_flag, device_mask_dict, action_space, param_rang_dict))

# Output
#
# OrderedDict([('IBP', '3.0u'), ('R0', '20.0k'), ('R1', '12.0k'), ('CM', '120.0p'), ('w_M13_per_finger',
# '4.0u'), ('l_M13', '3.0u'), ('nf_M13', '4'), ('w_M12_per_finger', '4.0u'), ('l_M12', '3.0u'), ('nf_M12', '4'),
# ('w_M11_per_finger', '4.0u'), ('l_M11', '3.0u'), ('nf_M11', '4'), ('w_M2_per_finger', '4.0u'), ('l_M2', '3.0u'),
# ('nf_M2', '4'), ('w_M10_per_finger', '8.5u'), ('l_M10', '4.0u'), ('nf_M10', '4'), ('w_M9_per_finger', '8.5u'),
# ('l_M9', '4.0u'), ('nf_M9', '4'), ('w_M6_per_finger', '4.5u'), ('l_M6', '1.0u'), ('nf_M6', '5'),
# ('w_M5_per_finger', '4.5u'), ('l_M5', '1.0u'), ('nf_M5', '5'), ('w_M4_per_finger', '7.0u'), ('l_M4', '9.5u'),
# ('nf_M4', '4'), ('w_M1_per_finger', '6.0u'), ('l_M1', '7.5u'), ('nf_M1', '4'), ('w_M0_per_finger', '300.0u'),
# ('l_M0', '0.5u'), ('nf_M0', '5')])
