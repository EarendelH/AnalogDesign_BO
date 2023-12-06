import numpy as np
import gymnasium
import yaml


def gen_action_space(agent_assign_yaml_path):
    """
    Generate action space for the environment.
    Parameters:
        agent_assign.yaml: the file that defines the action space including agent name and corresponding device name.
    Returns:
        MultiDiscrete: the action space for the multi-agent environment.
    """

    # Define operation:
    # 0: -1 index;
    # 1: Keep the same;
    # 2: +1 index;
    operation_number = 3

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    action_space_dict = {}
    for group_name, element in agent_assign.items():
        action_space_dict[group_name] = gymnasium.spaces.MultiDiscrete([operation_number] * len(element))

    return action_space_dict


# Test Code
agent_assign_yaml_path = "../config/agent_assign.yaml"
action_space_dict = gen_action_space(agent_assign_yaml_path)
print(action_space_dict)
