import numpy
from collections import OrderedDict
import copy


def masked_action_dict_mapping(device_mask_dict, step_action_dict):
    """
    Map the device mask dictionary to the step action dictionary.
    :param device_mask_dict: device mask dictionary
    :param step_action_dict: step action dictionary
    :return: mapped_step_action_dict: mapped step action dictionary
    """

    flattened_action_dict = {}
    for agent_key, agent_dict in step_action_dict.items():
        for key, value in agent_dict.items():
            if key in flattened_action_dict:
                raise ValueError(f'Duplicate key {key} found in device mask dictionary')
            flattened_action_dict[key] = value

    for key in device_mask_dict:
        if key in flattened_action_dict:
            new_values = device_mask_dict[key]
            for new_key in new_values:
                if new_key.endswith('_Match'):
                    match_action = flattened_action_dict[key][:2]
                    new_value = numpy.append(match_action, flattened_action_dict[new_key])
                else:
                    new_value = flattened_action_dict[key]
                flattened_action_dict[new_key] = new_value

    # Clean up keys
    mapped_action_dict = OrderedDict()
    for key, value in flattened_action_dict.items():
        new_key = key.removesuffix('_Match')
        mapped_action_dict[new_key] = value

    mapped_step_action_dict = copy.deepcopy(step_action_dict)

    # for key in device_mask_dict:
    #     for agent_key in step_action_dict:
    #         if key in step_action_dict[agent_key]:
    #             new_values = device_mask_dict[key]
    #             for new_key in new_values:
    #                 if new_key.endswith('_Match'):
    #                     match_action = step_action_dict[agent_key][key][:2]
    #                     new_value = numpy.append(match_action, step_action_dict[agent_key][new_key])
    #                 else:
    #                     new_value = step_action_dict[agent_key][key]
    #                 step_action_dict[agent_key][new_key] = new_value

    # mapped_action_dict = OrderedDict()
    # for agent, agent_dict in step_action_dict.items():
    #     new_agent_dict = OrderedDict()
    #     for key, value in agent_dict.items():
    #         if key.endswith('_Match'):
    #             new_key = key.replace('_Match', '')
    #         else:
    #             new_key = key

    #         new_agent_dict[new_key] = value
    #     mapped_action_dict[agent] = new_agent_dict
    # mapped_step_action_dict = copy.deepcopy(mapped_action_dict)
    return mapped_step_action_dict
