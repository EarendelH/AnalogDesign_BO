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
# agent_assign_yaml_path = "../config/agent_assign.yaml"
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
