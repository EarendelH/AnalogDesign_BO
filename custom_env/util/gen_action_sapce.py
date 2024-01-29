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

def gen_masked_action_space(action_mask_flag, device_mask_yaml_path, agent_assign_yaml_path):
    """
    Generate action space for the environment.
    Parameters:
        action_mask_flag: A flag to determine whether to apply the device mask.
        device_mask_yaml_path: Path to the YAML file that contains devices to be masked.
        agent_assign_yaml_path: Path to the YAML file that defines the action space including agent name and corresponding device name.
    Returns:
        gymnasium.spaces.Dict: The action space for the multi-agent environment as a dictionary of MultiDiscrete spaces.
    """

    # Define the number of operations for action space
    # 0: -1 index;
    # 1: Keep the same;
    # 2: +1 index;
    operation_number = 3

    # Load agent assignment configuration from YAML file
    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Load device mask configuration from YAML file
    with open(device_mask_yaml_path, 'r') as file:
        device_mask = yaml.safe_load(file)

    # Initialize the dictionary to store action spaces for each agent group
    action_space_dict = {}

    # Apply device mask if action_mask_flag is set to True
    if action_mask_flag:
        # Iterate through each device mask key and its values
        for key, values in device_mask.items():
            # Iterate through each agent group and its device list
            for group_name, device_list in agent_assign.items():
                # Check if the mask key is in the device list of the agent group
                if key in device_list:
                    # Iterate through the mask values and remove them from the device list if present
                    for value in values:
                        if value in device_list:
                            device_list.remove(value)

    # Generate the action space for each type of device
    for group_name, device_list in agent_assign.items():
        space_dict = {}
        # Iterate through each device in the device list
        for device in device_list:
            # Determine the action space based on the device type (e.g., MOSFET)
            if device.startswith('M'):
                space = gymnasium.spaces.MultiDiscrete([operation_number] * 3)
            else:
                space = gymnasium.spaces.MultiDiscrete([operation_number])
            space_dict[device] = space
        # Assign the action space for each agent group
        action_space_dict[group_name] = gymnasium.spaces.Dict(space_dict)

    # Return the complete action space as a dictionary
    return gymnasium.spaces.Dict(action_space_dict)

# Test Code
# agent_assign_yaml_path = "../config/agent_assign.yaml"
# device_mask_yaml_path = "../config/device_mask.yaml"
# action_space_dict = gen_masked_action_space(True, device_mask_yaml_path, agent_assign_yaml_path)
# print(action_space_dict)

# Output
# Dict('Agent_1': Dict('M13': MultiDiscrete([3 3 3]), 'M16': MultiDiscrete([3 3 3]), 'M23': MultiDiscrete([3 3
# 3]), 'M24': MultiDiscrete([3 3 3]), 'M25': MultiDiscrete([3 3 3]), 'M35': MultiDiscrete([3 3 3]),
# 'M36': MultiDiscrete([3 3 3])), 'Agent_2': Dict('M17': MultiDiscrete([3 3 3])), 'Agent_3': Dict('M12':
# MultiDiscrete([3 3 3]), 'M20': MultiDiscrete([3 3 3]), 'M22': MultiDiscrete([3 3 3])), 'Agent_4': Dict('IB':
# MultiDiscrete([3])))

# Test Code
# agent_assign_yaml_path = "../config/agent_assign.yaml"
# device_mask_yaml_path = "../config/device_mask.yaml"
# action_space_dict = gen_masked_action_space(False, device_mask_yaml_path, agent_assign_yaml_path)
# print(action_space_dict)

# Output
# Dict('Agent_1': Dict('M13': MultiDiscrete([3 3 3]), 'M14': MultiDiscrete([3 3 3]), 'M16': MultiDiscrete([3 3
# 3]), 'M23': MultiDiscrete([3 3 3]), 'M24': MultiDiscrete([3 3 3]), 'M25': MultiDiscrete([3 3 3]),
# 'M35': MultiDiscrete([3 3 3]), 'M36': MultiDiscrete([3 3 3])), 'Agent_2': Dict('M17': MultiDiscrete([3 3 3]),
# 'M18': MultiDiscrete([3 3 3])), 'Agent_3': Dict('M11': MultiDiscrete([3 3 3]), 'M12': MultiDiscrete([3 3 3]),
# 'M19': MultiDiscrete([3 3 3]), 'M20': MultiDiscrete([3 3 3]), 'M21': MultiDiscrete([3 3 3]), 'M22': MultiDiscrete([
# 3 3 3])), 'Agent_4': Dict('IB': MultiDiscrete([3])))
