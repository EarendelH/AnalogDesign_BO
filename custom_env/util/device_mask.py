import numpy
from collections import OrderedDict
import copy
from numpy import array


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
    # print(flattened_action_dict)

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
    # print(flattened_action_dict)

    # Clean up keys
    mapped_action_dict = OrderedDict()
    for key, value in flattened_action_dict.items():
        new_key = key.removesuffix('_Match')  # This works in Python 3.9+
        mapped_action_dict[new_key] = value

    mapped_step_action_dict = copy.deepcopy(mapped_action_dict)

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


# device_mask_dict = {'M9': ['M3_Match'], 'M10': ['M7_Match', 'M4_Match', 'M2_Match', 'M18_Match', 'M20_Match',
# 'M11_Match', 'M36_Match'], 'M1': ['M6_Match'], 'M16': ['M17'], 'M14': ['M15'], 'M37': ['M38'], 'M33': ['M35_Match',
# 'M34_Match', 'M13_Match'], 'M31': ['M32'], 'M27': ['M26'], 'M30': ['M29_Match'], 'M21': ['M22_Match'],
# 'M41': ['M42_Match'], 'M44': ['M43_Match', 'M45_Match'], 'M46': ['M47_Match']}
#
# step_action_dict = {'Agent_1':
# OrderedDict([('M0', array([0.76220273, 0.72228976, 0.29291678]))]), 'Agent_2': OrderedDict([('C2',
# array([0.26236587])), ('C3', array([0.32370724])), ('M11_Match', array([0.21498248])), ('M14', array([0.84177178,
# 0.33929708, 0.78271975])), ('M16', array([0.75796715, 0.3976568 , 0.1339106 ])), ('M18_Match',
# array([0.66487988])), ('M19', array([0.71292576, 0.48357929, 0.79139263])), ('M20_Match', array([0.7317345])),
# ('M6_Match', array([0.68512196]))]), 'Agent_3': OrderedDict([('C0', array([0.78025747])), ('M1', array([0.17870124,
# 0.85211795, 0.03094559])), ('M10', array([0.21854078, 0.83119063, 0.73553891])), ('M12', array([0.3569865 ,
# 0.8489258 , 0.11187358])), ('M2_Match', array([0.74411178])), ('M3_Match', array([0.01380882])), ('M4_Match',
# array([0.35064758])), ('M5', array([0.08065224, 0.62686755, 0.68299771])), ('M7_Match', array([0.40575266])),
# ('M8', array([0.71981535, 0.53748593, 0.02098471])), ('M9', array([0.7112846 , 0.90814671, 0.23835268])), ('R0',
# array([0.50987103]))]), 'Agent_4': OrderedDict([('C1', array([0.30273298])), ('C4', array([0.53775445])), ('C8',
# array([0.87958314])), ('M13_Match', array([0.15092034])), ('M21', array([0.68489274, 0.22022501, 0.53796338])),
# ('M22_Match', array([0.14293677])), ('M23', array([0.33120711, 0.86006291, 0.39337701])), ('M24',
# array([0.92745076, 0.34811023, 0.13888413])), ('M25', array([0.34033509, 0.67722572, 0.25072314])), ('M27',
# array([0.56235538, 0.34518786, 0.43647589])), ('M28', array([0.45430788, 0.16068788, 0.01184276])), ('M29_Match',
# array([0.20155146])), ('M30', array([0.75492242, 0.83067994, 0.39430279])), ('M31', array([0.6460649 , 0.64857765,
# 0.42196431])), ('M33', array([0.07406972, 0.189452  , 0.13832814])), ('M34_Match', array([0.03525872])),
# ('M35_Match', array([0.28856033])), ('M36_Match', array([0.85082825])), ('M37', array([0.34215285, 0.69180301,
# 0.21080258]))]), 'Agent_5': OrderedDict([('M40', array([0.82267292, 0.10140923, 0.04440631])), ('M41',
# array([0.45316946, 0.02501077, 0.09856557])), ('M42_Match', array([0.51817653])), ('M43_Match',
# array([0.1599782])), ('M44', array([0.06198189, 0.61399944, 0.53471361])), ('M45_Match', array([0.03208921])),
# ('M46', array([0.18230314, 0.83932209, 0.24484294])), ('M47_Match', array([0.66828285]))])}
#
# print(masked_action_dict_mapping(device_mask_dict, step_action_dict))
