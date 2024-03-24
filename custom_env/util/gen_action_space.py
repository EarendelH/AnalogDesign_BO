from collections import OrderedDict

import gymnasium
import yaml


def gen_action_space(agent_assign_yaml_path):
    """
    Generate action space for the environment.
    Parameters:
        agent_assign.yaml: the file that defines the action space including agent name and corresponding device name.
    Returns:
        Dict w/ Several MultiDiscrete: the action space for the multi-agent environment.
    """

    # Define operation:
    # 0: -1 index;
    # 1: Keep the same;
    # 2: +1 index;
    operation_number = 3

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    action_space_dict = {}
    # for group_name, element in agent_assign.items():
    #     action_space_dict[group_name] = gymnasium.spaces.MultiDiscrete([operation_number] * len(element))

    # For different type device, their action space is different. e.g., MOSFET -> MultiDiscrete([3]*3)
    for group_name, device_list in agent_assign.items():
        space_dict = {}
        for device in device_list:
            if device.startswith('M'):
                space = gymnasium.spaces.MultiDiscrete([operation_number] * 3)
            else:
                space = gymnasium.spaces.MultiDiscrete([operation_number])
            space_dict[device] = space
        action_space_dict[group_name] = gymnasium.spaces.Dict(space_dict)
    return gymnasium.spaces.Dict(action_space_dict)


# Test Code
# agent_assign_yaml_path = "../config_3/agent_assign.yaml"
# action_space_dict = gen_action_space(agent_assign_yaml_path)
# print(action_space_dict)
#
# all_action = {}
# for group_name, action_space in action_space_dict.items():
#     action = action_space_dict[group_name].sample()
#     all_action[group_name] = action
# print(all_action)
#
# all_action_flatten = OrderedDict()
# for action in all_action.values():
#     all_action_flatten.update(action)
# print(all_action_flatten)

# Output Dict('Agent_1': Dict('M13': MultiDiscrete([3 3 3]), 'M14': MultiDiscrete([3 3 3]), 'M16': MultiDiscrete([3 3
# 3]), 'M23': MultiDiscrete([3 3 3]), 'M24': MultiDiscrete([3 3 3]), 'M25': MultiDiscrete([3 3 3]),
# 'M35': MultiDiscrete([3 3 3]), 'M36': MultiDiscrete([3 3 3])), 'Agent_2': Dict('M17': MultiDiscrete([3 3 3]),
# 'M18': MultiDiscrete([3 3 3])), 'Agent_3': Dict('M11': MultiDiscrete([3 3 3]), 'M12': MultiDiscrete([3 3 3]),
# 'M19': MultiDiscrete([3 3 3]), 'M20': MultiDiscrete([3 3 3]), 'M21': MultiDiscrete([3 3 3]), 'M22': MultiDiscrete([
# 3 3 3])), 'Agent_4': Dict('IB': MultiDiscrete([3])))
#
# {'Agent_1': OrderedDict([('M13', array([1, 2, 2])), ('M14', array([2, 1, 1])), ('M16', array([1, 2, 1])), ('M23',
# array([2, 0, 2])), ('M24', array([2, 0, 1])), ('M25', array([2, 2, 2])), ('M35', array([0, 0, 2])), ('M36',
# array([0, 1, 2]))]), 'Agent_2': OrderedDict([('M17', array([0, 0, 1])), ('M18', array([1, 0, 1]))]), 'Agent_3':
# OrderedDict([('M11', array([1, 0, 0])), ('M12', array([0, 2, 0])), ('M19', array([0, 2, 0])), ('M20', array([2, 0,
# 1])), ('M21', array([2, 2, 0])), ('M22', array([1, 2, 2]))]), 'Agent_4': OrderedDict([('IB', array([1]))])}
#
# OrderedDict([('M13', array([1, 2, 2])), ('M14', array([2, 1, 1])), ('M16', array([1, 2, 1])), ('M23', array([2, 0,
# 2])), ('M24', array([2, 0, 1])), ('M25', array([2, 2, 2])), ('M35', array([0, 0, 2])), ('M36', array([0, 1, 2])),
# ('M17', array([0, 0, 1])), ('M18', array([1, 0, 1])), ('M11', array([1, 0, 0])), ('M12', array([0, 2, 0])), ('M19',
# array([0, 2, 0])), ('M20', array([2, 0, 1])), ('M21', array([2, 2, 0])), ('M22', array([1, 2, 2])), ('IB',
# array([1]))])

def gen_masked_action_space(action_mask_flag, device_mask_dict, agent_assign_dict):
    """
    Generate action space for the environment.
    Parameters:
        action_mask_flag: A flag to determine whether to apply the device mask.
        device_mask_dict: dict to the YAML file that contains devices to be masked.
        agent_assign_dict: dict to the YAML file that defines the action space including agent name and corresponding device name.
    Returns:
        gymnasium.spaces.Dict: The action space for the multi-agent environment as a dictionary of MultiDiscrete spaces.
    """

    # Define the number of operations for action space
    # 0: -1 index;
    # 1: Keep the same;
    # 2: +1 index;
    operation_number = 3  # Define the number of operations for action space

    # Apply device mask if flagged
    if action_mask_flag:
        for key, values in device_mask_dict.items():
            for device_list in agent_assign_dict.values():
                device_list[:] = [device for device in device_list if device not in values]

    # Generate action spaces
    action_space_dict = {
        group_name: gymnasium.spaces.Dict({
            device: gymnasium.spaces.MultiDiscrete([operation_number] * (3 if device.startswith('M') else 1))
            for device in device_list
        })
        for group_name, device_list in agent_assign_dict.items()
    }

    return gymnasium.spaces.Dict(action_space_dict)


# Test Code
# agent_assign_yaml_path = "../config/agent_assign.yaml"
# with open(agent_assign_yaml_path, 'r') as file:
#     agent_assign_dict = yaml.safe_load(file)
# device_mask_yaml_path = "../config/device_mask.yaml"
# with open(device_mask_yaml_path, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# action_space_dict = gen_masked_action_space(True, device_mask_dict, agent_assign_dict)
# print(action_space_dict)

# Output

# Dict('Agent_1': Dict('IBP': MultiDiscrete([3]), 'M1': MultiDiscrete([3 3 3]), 'M4': MultiDiscrete([3 3 3]),
# 'M5': MultiDiscrete([3 3 3])), 'Agent_2': Dict('M9': MultiDiscrete([3 3 3])), 'Agent_3': Dict('M12': MultiDiscrete(
# [3 3 3])), 'Agent_4': Dict('CM': MultiDiscrete([3]), 'M0': MultiDiscrete([3 3 3]), 'R0': MultiDiscrete([3]),
# 'R1': MultiDiscrete([3])))

# Test Code
# agent_assign_yaml_path = "../config/agent_assign.yaml"
# with open(agent_assign_yaml_path, 'r') as file:
#     agent_assign_dict = yaml.safe_load(file)
# device_mask_yaml_path = "../config/device_mask.yaml"
# with open(device_mask_yaml_path, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# action_space_dict = gen_masked_action_space(False, device_mask_dict, agent_assign_dict)
# print(action_space_dict)


# Output

# Dict('Agent_1': Dict('IBP': MultiDiscrete([3]), 'M1': MultiDiscrete([3 3 3]), 'M4': MultiDiscrete([3 3 3]),
# 'M5': MultiDiscrete([3 3 3]), 'M6': MultiDiscrete([3 3 3])), 'Agent_2': Dict('M10': MultiDiscrete([3 3 3]),
# 'M9': MultiDiscrete([3 3 3])), 'Agent_3': Dict('M11': MultiDiscrete([3 3 3]), 'M12': MultiDiscrete([3 3 3]),
# 'M13': MultiDiscrete([3 3 3]), 'M2': MultiDiscrete([3 3 3])), 'Agent_4': Dict('CM': MultiDiscrete([3]),
# 'M0': MultiDiscrete([3 3 3]), 'R0': MultiDiscrete([3]), 'R1': MultiDiscrete([3])))


def gen_masked_continuous_action_space(action_mask_flag, device_mask_dict, agent_assign_dict):
    """
    Generate action space for the environment.
    Parameters:
        action_mask_flag: A flag to determine whether to apply the device mask.
        device_mask_dict: dict to the YAML file that contains devices to be masked.
        agent_assign_dict: dict to the YAML file that defines the action space including agent name and corresponding device name.
    Returns:
        gymnasium.spaces.Dict: The action space for the multi-agent environment as a dictionary of MultiDiscrete spaces.
    """

    single_mos_act_space = gymnasium.spaces.Box(low=0, high=1, shape=(3,), dtype=float)
    single_act_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=float)

    # Apply device mask if flagged
    if action_mask_flag:
        for key, values in device_mask_dict.items():
            for device_list in agent_assign_dict.values():
                device_list[:] = [device for device in device_list if device not in values]

    # Generate action spaces
    continuous_action_space_dict = {
        group_name: gymnasium.spaces.Dict({
            device: single_mos_act_space if device.startswith('M') else single_act_space
            for device in device_list
        })
        for group_name, device_list in agent_assign_dict.items()
    }

    return gymnasium.spaces.Dict(continuous_action_space_dict)


# Test Code

# agent_assign_yaml_path = "../config/agent_assign.yaml"
# with open(agent_assign_yaml_path, 'r') as file:
#     agent_assign_dict = yaml.safe_load(file)
# device_mask_yaml_path = "../config/device_mask.yaml"
# with open(device_mask_yaml_path, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# action_space_dict = gen_masked_continuous_action_space(True, device_mask_dict, agent_assign_dict)
# print(action_space_dict)
# print(action_space_dict.sample())

# Output

# Dict('Agent_1': Dict('IBP': Box(0.0, 1.0, (1,), float64), 'M1': Box(0.0, 1.0, (3,), float64), 'M4': Box(0.0, 1.0,
# (3,), float64), 'M5': Box(0.0, 1.0, (3,), float64)), 'Agent_2': Dict('M9': Box(0.0, 1.0, (3,), float64)),
# 'Agent_3': Dict('M12': Box(0.0, 1.0, (3,), float64)), 'Agent_4': Dict('CM': Box(0.0, 1.0, (1,), float64),
# 'M0': Box(0.0, 1.0, (3,), float64), 'R0': Box(0.0, 1.0, (1,), float64), 'R1': Box(0.0, 1.0, (1,), float64)))

# OrderedDict([('Agent_1', OrderedDict([('IBP', array([0.58232024])), ('M1', array([0.55986021, 0.71259075,
# 0.32580201])), ('M4', array([0.66283334, 0.9521516 , 0.37748751])), ('M5', array([0.40918823, 0.07680309,
# 0.44814611]))])), ('Agent_2', OrderedDict([('M9', array([0.82469542, 0.35882003, 0.36313247]))])), ('Agent_3',
# OrderedDict([('M12', array([0.37167306, 0.25930124, 0.37495792]))])), ('Agent_4', OrderedDict([('CM',
# array([0.54670056])), ('M0', array([0.54749156, 0.13308131, 0.40636367])), ('R0', array([0.61302462])), ('R1',
# array([0.37650811]))]))])
