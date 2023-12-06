import numpy as np
import gymnasium
import yaml


def gen_action_space(agent_assign_yaml_path):
    """
    Generate action space for the environment.
    Parameters:
        agent_assign.yaml: the file that defines the action space including agent name and corresponding device name.
    Returns:
        Tuple w/ Several MultiDiscrete: the action space for the multi-agent environment.
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
        space_list = []
        for device in device_list:
            if device.startswith('M'):
                space = gymnasium.spaces.MultiDiscrete([operation_number] * 3)
            else:
                space = gymnasium.spaces.MultiDiscrete([operation_number])
            space_list.append(space)
        action_space_dict[group_name] = gymnasium.spaces.Tuple(space_list)
    return action_space_dict


# Test Code
agent_assign_yaml_path = "../config/agent_assign.yaml"
action_space_dict = gen_action_space(agent_assign_yaml_path)
print(action_space_dict)

# Output {'Group_1': Tuple(MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3
# 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3])), 'Group_2':
# Tuple(MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3])), 'Group_3': Tuple(MultiDiscrete([3 3 3]), MultiDiscrete([3 3
# 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3]), MultiDiscrete([3 3 3])), 'Group_4':
# Tuple(MultiDiscrete([3]))}
