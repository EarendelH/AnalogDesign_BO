import gymnasium
import numpy as np
import yaml

from util.util_func import unit_conversion


def gen_obs_space(result_config_file, param_range_config_file):
    """
    Generate observation space for the custom environment.
    :param result_config_file: path of the result config file
    :param param_range_config_file: path of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    # Import YAML file
    with open(result_config_file, 'r') as file:
        result_config = yaml.safe_load(file)
    with open(param_range_config_file, 'r') as file:
        param_range_config = yaml.safe_load(file)

    # Create spaces for ideal_specs and cur_specs
    ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                          for key in sum(result_config.values(), [])}
    cur_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        for key in sum(result_config.values(), [])}

    # Create spaces for cur_param
    cur_param_spaces = {}
    for component, data in param_range_config.items():
        for param in data['params']:
            variable_name = param['variable_name']
            range_min, range_max = param['value']['range']
            cur_param_spaces[variable_name] = gymnasium.spaces.Box(low=unit_conversion(range_min),
                                                                   high=unit_conversion(range_max), shape=(1,),
                                                                   dtype=np.float32)

    # Combine three dicts into one gymnasium.spaces.Dict
    obs_space = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
        'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
        'cur_param': gymnasium.spaces.Dict(cur_param_spaces)
    })

    return obs_space


def gen_obs_space_simple(result_config_file, param_range_config_file, agent_assign_yaml_path):
    """
    Generate observation space for the custom environment.
    :param result_config_file: path of the result config file
    :param param_range_config_file: path of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Import YAML file
    with open(result_config_file, 'r') as file:
        result_config = yaml.safe_load(file)
    with open(param_range_config_file, 'r') as file:
        param_range_config = yaml.safe_load(file)

    # Create spaces for ideal_specs and cur_specs
    ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                          for key in sum(result_config.values(), [])}
    cur_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        for key in sum(result_config.values(), [])}

    # Create spaces for cur_param
    cur_param_spaces = {}
    for component, data in param_range_config.items():
        for param in data['params']:
            variable_name = param['variable_name']
            range_min, range_max = param['value']['range']
            cur_param_spaces[variable_name] = gymnasium.spaces.Box(low=unit_conversion(range_min),
                                                                   high=unit_conversion(range_max), shape=(1,),
                                                                   dtype=np.float32)

    # Combine three dicts into one gymnasium.spaces.Dict
    obs_space_single = gymnasium.spaces.Dict({
        'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces)
    })

    obs_space = gymnasium.spaces.Dict()
    for group_name in agent_assign.keys():
        obs_space[group_name] = obs_space_single

    return obs_space


# Test Code
# result_config_file = "../config/result.yaml"
# param_range_config_file = "../config/param_range.yaml"
# observation_space = gen_obs_space(result_config_file, param_range_config_file)
# print(f"observation_space: {observation_space}")
#
# observation_space_sample = observation_space.sample()
# print(f"observation_space_sample: {observation_space_sample}")

# Output
# observation_space: Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,), float32), 'l_M11': Box(5e-07, 1e-05,
# (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07, 1e-05, (1,), float32), 'l_M14': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32), 'l_M17': Box(5e-07, 1e-05, (1,), float32),
# 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05, (1,), float32), 'l_M20': Box(5e-07, 1e-05,
# (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(5e-07, 1e-05, (1,), float32), 'l_M23': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32), 'l_M25': Box(5e-07, 1e-05, (1,), float32),
# 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05, (1,), float32), 'nf_M11': Box(1.0, 20.0, (1,
# ), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(1.0, 20.0, (1,), float32), 'nf_M14': Box(1.0,
# 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32), 'nf_M17': Box(1.0, 20.0, (1,), float32),
# 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,), float32), 'nf_M20': Box(1.0, 20.0, (1,),
# float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0, 20.0, (1,), float32), 'nf_M23': Box(1.0,
# 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32), 'nf_M25': Box(1.0, 20.0, (1,), float32),
# 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,), float32), 'w_M11_per_finger': Box(5e-07,
# 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M13_per_finger': Box(5e-07, 1e-05,
# (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M16_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M18_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M20_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M22_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M24_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M35_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,), float32)), 'cur_specs': Dict('gainBandWidth': Box(0.0, 1.0,
# (1,), float32), 'phaseMargin': Box(0.0, 1.0, (1,), float32), 'powerSupplyRejectionRatio': Box(0.0, 1.0, (1,),
# float32), 'pwr': Box(0.0, 1.0, (1,), float32), 'slewRateDown': Box(0.0, 1.0, (1,), float32), 'slewRateUp': Box(0.0,
# 1.0, (1,), float32)), 'ideal_specs': Dict('gainBandWidth': Box(0.0, 1.0, (1,), float32), 'phaseMargin': Box(0.0,
# 1.0, (1,), float32), 'powerSupplyRejectionRatio': Box(0.0, 1.0, (1,), float32), 'pwr': Box(0.0, 1.0, (1,),
# float32), 'slewRateDown': Box(0.0, 1.0, (1,), float32), 'slewRateUp': Box(0.0, 1.0, (1,), float32)))

# observation_space_sample: OrderedDict([('cur_param', OrderedDict([('IB', array([2.7066786e-05], dtype=float32)),
# ('l_M11', array([7.069547e-07], dtype=float32)), ('l_M12', array([4.4842827e-06], dtype=float32)), ('l_M13',
# array([3.9931497e-06], dtype=float32)), ('l_M14', array([6.3652885e-07], dtype=float32)), ('l_M16',
# array([8.0275995e-06], dtype=float32)), ('l_M17', array([5.8787855e-06], dtype=float32)), ('l_M18',
# array([7.584472e-07], dtype=float32)), ('l_M19', array([2.2041734e-06], dtype=float32)), ('l_M20',
# array([5.0734566e-06], dtype=float32)), ('l_M21', array([3.5145003e-06], dtype=float32)), ('l_M22',
# array([9.0281355e-06], dtype=float32)), ('l_M23', array([6.743321e-06], dtype=float32)), ('l_M24',
# array([1.6829325e-06], dtype=float32)), ('l_M25', array([9.29393e-06], dtype=float32)), ('l_M35',
# array([1.7327176e-06], dtype=float32)), ('l_M36', array([7.5696485e-06], dtype=float32)), ('nf_M11',
# array([8.507872], dtype=float32)), ('nf_M12', array([17.056787], dtype=float32)), ('nf_M13', array([8.334092],
# dtype=float32)), ('nf_M14', array([9.521806], dtype=float32)), ('nf_M16', array([16.598408], dtype=float32)),
# ('nf_M17', array([17.73528], dtype=float32)), ('nf_M18', array([6.257158], dtype=float32)), ('nf_M19',
# array([4.8691764], dtype=float32)), ('nf_M20', array([11.576777], dtype=float32)), ('nf_M21', array([17.770025],
# dtype=float32)), ('nf_M22', array([13.11965], dtype=float32)), ('nf_M23', array([4.5762987], dtype=float32)),
# ('nf_M24', array([5.087942], dtype=float32)), ('nf_M25', array([4.5993686], dtype=float32)), ('nf_M35',
# array([8.574895], dtype=float32)), ('nf_M36', array([6.6746016], dtype=float32)), ('w_M11_per_finger',
# array([5.2349124e-06], dtype=float32)), ('w_M12_per_finger', array([5.255114e-06], dtype=float32)),
# ('w_M13_per_finger', array([9.2266873e-07], dtype=float32)), ('w_M14_per_finger', array([1.1110886e-06],
# dtype=float32)), ('w_M16_per_finger', array([4.9816827e-06], dtype=float32)), ('w_M17_per_finger',
# array([4.9746354e-06], dtype=float32)), ('w_M18_per_finger', array([3.792815e-06], dtype=float32)),
# ('w_M19_per_finger', array([2.269577e-06], dtype=float32)), ('w_M20_per_finger', array([6.874215e-06],
# dtype=float32)), ('w_M21_per_finger', array([9.167475e-06], dtype=float32)), ('w_M22_per_finger',
# array([3.5721155e-06], dtype=float32)), ('w_M23_per_finger', array([5.4332527e-06], dtype=float32)),
# ('w_M24_per_finger', array([9.864257e-07], dtype=float32)), ('w_M25_per_finger', array([9.3203903e-07],
# dtype=float32)), ('w_M35_per_finger', array([5.4637517e-06], dtype=float32)), ('w_M36_per_finger',
# array([8.212538e-06], dtype=float32))])), ('cur_specs', OrderedDict([('gainBandWidth', array([0.2701608],
# dtype=float32)), ('phaseMargin', array([0.16946422], dtype=float32)), ('powerSupplyRejectionRatio',
# array([0.40048453], dtype=float32)), ('pwr', array([0.66404974], dtype=float32)), ('slewRateDown',
# array([0.28164378], dtype=float32)), ('slewRateUp', array([0.99810416], dtype=float32))])), ('ideal_specs',
# OrderedDict([('gainBandWidth', array([0.8282451], dtype=float32)), ('phaseMargin', array([0.07822091],
# dtype=float32)), ('powerSupplyRejectionRatio', array([0.2896765], dtype=float32)), ('pwr', array([0.8886158],
# dtype=float32)), ('slewRateDown', array([0.6193389], dtype=float32)), ('slewRateUp', array([0.75055516],
# dtype=float32))]))])

def gen_obs_space_extend(result_config_file, param_range_config_file, agent_assign_yaml_path):
    """
    Generate observation space for the custom environment.
    :param result_config_file: path of the result config file
    :param param_range_config_file: path of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Import YAML file
    with open(result_config_file, 'r') as file:
        result_config = yaml.safe_load(file)
    with open(param_range_config_file, 'r') as file:
        param_range_config = yaml.safe_load(file)

    # Create spaces for ideal_specs and cur_specs
    ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                          for key in sum(result_config.values(), [])}
    cur_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        for key in sum(result_config.values(), [])}

    # Create spaces for cur_param
    cur_param_spaces = {}
    for component, data in param_range_config.items():
        for param in data['params']:
            variable_name = param['variable_name']
            range_min, range_max = param['value']['range']
            cur_param_spaces[variable_name] = gymnasium.spaces.Box(low=unit_conversion(range_min),
                                                                   high=unit_conversion(range_max), shape=(1,),
                                                                   dtype=np.float32)

    # Combine three dicts into one gymnasium.spaces.Dict
    obs_space_single = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
        'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
        'cur_param': gymnasium.spaces.Dict(cur_param_spaces)
    })

    obs_space = gymnasium.spaces.Dict()
    for group_name in agent_assign.keys():
        obs_space[group_name] = obs_space_single

    return obs_space

# reset_obs: {'Agent_1': OrderedDict([('cur_specs', {'pwr': array([0.00020386], dtype=float32), 'phaseMargin': array(
# [63.3612], dtype=float32), 'gainBandWidth': array([1.4186192e+08], dtype=float32), 'slewRateUp': array([
# 1025826.06], dtype=float32), 'slewRateDown': array([1064081.5], dtype=float32), 'powerSupplyRejectionRatio': array(
# [48.528435], dtype=float32)}), ('ideal_specs', {'gainBandWidth': array([9.199741e+08], dtype=float32),
# 'phaseMargin': array([74.04052], dtype=float32), 'powerSupplyRejectionRatio': array([92.782036], dtype=float32),
# 'pwr': array([0.00012023], dtype=float32), 'slewRateDown': array([6131225.5], dtype=float32), 'slewRateUp': array([
# 4413834.5], dtype=float32)}), ('cur_param', {'w_M14_per_finger': array([5.e-06], dtype=float32), 'l_M14': array([
# 5.e-06], dtype=float32), 'nf_M14': array([10.], dtype=float32), 'w_M35_per_finger': array([5.e-06], dtype=float32),
# 'l_M35': array([5.e-06], dtype=float32), 'nf_M35': array([10.], dtype=float32), 'w_M25_per_finger': array([5.e-06],
# dtype=float32), 'l_M25': array([5.e-06], dtype=float32), 'nf_M25': array([10.], dtype=float32), 'w_M13_per_finger':
# array([5.e-06], dtype=float32), 'l_M13': array([5.e-06], dtype=float32), 'nf_M13': array([10.], dtype=float32),
# 'w_M12_per_finger': array([5.e-06], dtype=float32), 'l_M12': array([5.e-06], dtype=float32), 'nf_M12': array([10.],
# dtype=float32), 'w_M11_per_finger': array([5.e-06], dtype=float32), 'l_M11': array([5.e-06], dtype=float32),
# 'nf_M11': array([10.], dtype=float32), 'w_M20_per_finger': array([5.e-06], dtype=float32), 'l_M20': array([5.e-06],
# dtype=float32), 'nf_M20': array([10.], dtype=float32), 'w_M19_per_finger': array([5.e-06], dtype=float32),
# 'l_M19': array([5.e-06], dtype=float32), 'nf_M19': array([10.], dtype=float32), 'w_M36_per_finger': array([5.e-06],
# dtype=float32), 'l_M36': array([5.e-06], dtype=float32), 'nf_M36': array([10.], dtype=float32), 'w_M16_per_finger':
# array([5.e-06], dtype=float32), 'l_M16': array([5.e-06], dtype=float32), 'nf_M16': array([10.], dtype=float32),
# 'w_M24_per_finger': array([5.e-06], dtype=float32), 'l_M24': array([5.e-06], dtype=float32), 'nf_M24': array([10.],
# dtype=float32), 'w_M23_per_finger': array([5.e-06], dtype=float32), 'l_M23': array([5.e-06], dtype=float32),
# 'nf_M23': array([10.], dtype=float32), 'w_M22_per_finger': array([5.e-06], dtype=float32), 'l_M22': array([5.e-06],
# dtype=float32), 'nf_M22': array([10.], dtype=float32), 'w_M21_per_finger': array([5.e-06], dtype=float32),
# 'l_M21': array([5.e-06], dtype=float32), 'nf_M21': array([10.], dtype=float32), 'w_M18_per_finger': array([5.e-06],
# dtype=float32), 'l_M18': array([5.e-06], dtype=float32), 'nf_M18': array([10.], dtype=float32), 'w_M17_per_finger':
# array([5.e-06], dtype=float32), 'l_M17': array([5.e-06], dtype=float32), 'nf_M17': array([10.], dtype=float32),
# 'IB': array([2.5e-05], dtype=float32)})]), 'Agent_2': OrderedDict([('cur_specs', {'pwr': array([0.00020386],
# dtype=float32), 'phaseMargin': array([63.3612], dtype=float32), 'gainBandWidth': array([1.4186192e+08],
# dtype=float32), 'slewRateUp': array([1025826.06], dtype=float32), 'slewRateDown': array([1064081.5],
# dtype=float32), 'powerSupplyRejectionRatio': array([48.528435], dtype=float32)}), ('ideal_specs', {'gainBandWidth':
# array([9.199741e+08], dtype=float32), 'phaseMargin': array([74.04052], dtype=float32), 'powerSupplyRejectionRatio':
# array([92.782036], dtype=float32), 'pwr': array([0.00012023], dtype=float32), 'slewRateDown': array([6131225.5],
# dtype=float32), 'slewRateUp': array([4413834.5], dtype=float32)}), ('cur_param', {'w_M14_per_finger': array([
# 5.e-06], dtype=float32), 'l_M14': array([5.e-06], dtype=float32), 'nf_M14': array([10.], dtype=float32),
# 'w_M35_per_finger': array([5.e-06], dtype=float32), 'l_M35': array([5.e-06], dtype=float32), 'nf_M35': array([10.],
# dtype=float32), 'w_M25_per_finger': array([5.e-06], dtype=float32), 'l_M25': array([5.e-06], dtype=float32),
# 'nf_M25': array([10.], dtype=float32), 'w_M13_per_finger': array([5.e-06], dtype=float32), 'l_M13': array([5.e-06],
# dtype=float32), 'nf_M13': array([10.], dtype=float32), 'w_M12_per_finger': array([5.e-06], dtype=float32),
# 'l_M12': array([5.e-06], dtype=float32), 'nf_M12': array([10.], dtype=float32), 'w_M11_per_finger': array([5.e-06],
# dtype=float32), 'l_M11': array([5.e-06], dtype=float32), 'nf_M11': array([10.], dtype=float32), 'w_M20_per_finger':
# array([5.e-06], dtype=float32), 'l_M20': array([5.e-06], dtype=float32), 'nf_M20': array([10.], dtype=float32),
# 'w_M19_per_finger': array([5.e-06], dtype=float32), 'l_M19': array([5.e-06], dtype=float32), 'nf_M19': array([10.],
# dtype=float32), 'w_M36_per_finger': array([5.e-06], dtype=float32), 'l_M36': array([5.e-06], dtype=float32),
# 'nf_M36': array([10.], dtype=float32), 'w_M16_per_finger': array([5.e-06], dtype=float32), 'l_M16': array([5.e-06],
# dtype=float32), 'nf_M16': array([10.], dtype=float32), 'w_M24_per_finger': array([5.e-06], dtype=float32),
# 'l_M24': array([5.e-06], dtype=float32), 'nf_M24': array([10.], dtype=float32), 'w_M23_per_finger': array([5.e-06],
# dtype=float32), 'l_M23': array([5.e-06], dtype=float32), 'nf_M23': array([10.], dtype=float32), 'w_M22_per_finger':
# array([5.e-06], dtype=float32), 'l_M22': array([5.e-06], dtype=float32), 'nf_M22': array([10.], dtype=float32),
# 'w_M21_per_finger': array([5.e-06], dtype=float32), 'l_M21': array([5.e-06], dtype=float32), 'nf_M21': array([10.],
# dtype=float32), 'w_M18_per_finger': array([5.e-06], dtype=float32), 'l_M18': array([5.e-06], dtype=float32),
# 'nf_M18': array([10.], dtype=float32), 'w_M17_per_finger': array([5.e-06], dtype=float32), 'l_M17': array([5.e-06],
# dtype=float32), 'nf_M17': array([10.], dtype=float32), 'IB': array([2.5e-05], dtype=float32)})]), 'Agent_3':
# OrderedDict([('cur_specs', {'pwr': array([0.00020386], dtype=float32), 'phaseMargin': array([63.3612],
# dtype=float32), 'gainBandWidth': array([1.4186192e+08], dtype=float32), 'slewRateUp': array([1025826.06],
# dtype=float32), 'slewRateDown': array([1064081.5], dtype=float32), 'powerSupplyRejectionRatio': array([48.528435],
# dtype=float32)}), ('ideal_specs', {'gainBandWidth': array([9.199741e+08], dtype=float32), 'phaseMargin': array([
# 74.04052], dtype=float32), 'powerSupplyRejectionRatio': array([92.782036], dtype=float32), 'pwr': array([
# 0.00012023], dtype=float32), 'slewRateDown': array([6131225.5], dtype=float32), 'slewRateUp': array([4413834.5],
# dtype=float32)}), ('cur_param', {'w_M14_per_finger': array([5.e-06], dtype=float32), 'l_M14': array([5.e-06],
# dtype=float32), 'nf_M14': array([10.], dtype=float32), 'w_M35_per_finger': array([5.e-06], dtype=float32),
# 'l_M35': array([5.e-06], dtype=float32), 'nf_M35': array([10.], dtype=float32), 'w_M25_per_finger': array([5.e-06],
# dtype=float32), 'l_M25': array([5.e-06], dtype=float32), 'nf_M25': array([10.], dtype=float32), 'w_M13_per_finger':
# array([5.e-06], dtype=float32), 'l_M13': array([5.e-06], dtype=float32), 'nf_M13': array([10.], dtype=float32),
# 'w_M12_per_finger': array([5.e-06], dtype=float32), 'l_M12': array([5.e-06], dtype=float32), 'nf_M12': array([10.],
# dtype=float32), 'w_M11_per_finger': array([5.e-06], dtype=float32), 'l_M11': array([5.e-06], dtype=float32),
# 'nf_M11': array([10.], dtype=float32), 'w_M20_per_finger': array([5.e-06], dtype=float32), 'l_M20': array([5.e-06],
# dtype=float32), 'nf_M20': array([10.], dtype=float32), 'w_M19_per_finger': array([5.e-06], dtype=float32),
# 'l_M19': array([5.e-06], dtype=float32), 'nf_M19': array([10.], dtype=float32), 'w_M36_per_finger': array([5.e-06],
# dtype=float32), 'l_M36': array([5.e-06], dtype=float32), 'nf_M36': array([10.], dtype=float32), 'w_M16_per_finger':
# array([5.e-06], dtype=float32), 'l_M16': array([5.e-06], dtype=float32), 'nf_M16': array([10.], dtype=float32),
# 'w_M24_per_finger': array([5.e-06], dtype=float32), 'l_M24': array([5.e-06], dtype=float32), 'nf_M24': array([10.],
# dtype=float32), 'w_M23_per_finger': array([5.e-06], dtype=float32), 'l_M23': array([5.e-06], dtype=float32),
# 'nf_M23': array([10.], dtype=float32), 'w_M22_per_finger': array([5.e-06], dtype=float32), 'l_M22': array([5.e-06],
# dtype=float32), 'nf_M22': array([10.], dtype=float32), 'w_M21_per_finger': array([5.e-06], dtype=float32),
# 'l_M21': array([5.e-06], dtype=float32), 'nf_M21': array([10.], dtype=float32), 'w_M18_per_finger': array([5.e-06],
# dtype=float32), 'l_M18': array([5.e-06], dtype=float32), 'nf_M18': array([10.], dtype=float32), 'w_M17_per_finger':
# array([5.e-06], dtype=float32), 'l_M17': array([5.e-06], dtype=float32), 'nf_M17': array([10.], dtype=float32),
# 'IB': array([2.5e-05], dtype=float32)})]), 'Agent_4': OrderedDict([('cur_specs', {'pwr': array([0.00020386],
# dtype=float32), 'phaseMargin': array([63.3612], dtype=float32), 'gainBandWidth': array([1.4186192e+08],
# dtype=float32), 'slewRateUp': array([1025826.06], dtype=float32), 'slewRateDown': array([1064081.5],
# dtype=float32), 'powerSupplyRejectionRatio': array([48.528435], dtype=float32)}), ('ideal_specs', {'gainBandWidth':
# array([9.199741e+08], dtype=float32), 'phaseMargin': array([74.04052], dtype=float32), 'powerSupplyRejectionRatio':
# array([92.782036], dtype=float32), 'pwr': array([0.00012023], dtype=float32), 'slewRateDown': array([6131225.5],
# dtype=float32), 'slewRateUp': array([4413834.5], dtype=float32)}), ('cur_param', {'w_M14_per_finger': array([
# 5.e-06], dtype=float32), 'l_M14': array([5.e-06], dtype=float32), 'nf_M14': array([10.], dtype=float32),
# 'w_M35_per_finger': array([5.e-06], dtype=float32), 'l_M35': array([5.e-06], dtype=float32), 'nf_M35': array([10.],
# dtype=float32), 'w_M25_per_finger': array([5.e-06], dtype=float32), 'l_M25': array([5.e-06], dtype=float32),
# 'nf_M25': array([10.], dtype=float32), 'w_M13_per_finger': array([5.e-06], dtype=float32), 'l_M13': array([5.e-06],
# dtype=float32), 'nf_M13': array([10.], dtype=float32), 'w_M12_per_finger': array([5.e-06], dtype=float32),
# 'l_M12': array([5.e-06], dtype=float32), 'nf_M12': array([10.], dtype=float32), 'w_M11_per_finger': array([5.e-06],
# dtype=float32), 'l_M11': array([5.e-06], dtype=float32), 'nf_M11': array([10.], dtype=float32), 'w_M20_per_finger':
# array([5.e-06], dtype=float32), 'l_M20': array([5.e-06], dtype=float32), 'nf_M20': array([10.], dtype=float32),
# 'w_M19_per_finger': array([5.e-06], dtype=float32), 'l_M19': array([5.e-06], dtype=float32), 'nf_M19': array([10.],
# dtype=float32), 'w_M36_per_finger': array([5.e-06], dtype=float32), 'l_M36': array([5.e-06], dtype=float32),
# 'nf_M36': array([10.], dtype=float32), 'w_M16_per_finger': array([5.e-06], dtype=float32), 'l_M16': array([5.e-06],
# dtype=float32), 'nf_M16': array([10.], dtype=float32), 'w_M24_per_finger': array([5.e-06], dtype=float32),
# 'l_M24': array([5.e-06], dtype=float32), 'nf_M24': array([10.], dtype=float32), 'w_M23_per_finger': array([5.e-06],
# dtype=float32), 'l_M23': array([5.e-06], dtype=float32), 'nf_M23': array([10.], dtype=float32), 'w_M22_per_finger':
# array([5.e-06], dtype=float32), 'l_M22': array([5.e-06], dtype=float32), 'nf_M22': array([10.], dtype=float32),
# 'w_M21_per_finger': array([5.e-06], dtype=float32), 'l_M21': array([5.e-06], dtype=float32), 'nf_M21': array([10.],
# dtype=float32), 'w_M18_per_finger': array([5.e-06], dtype=float32), 'l_M18': array([5.e-06], dtype=float32),
# 'nf_M18': array([10.], dtype=float32), 'w_M17_per_finger': array([5.e-06], dtype=float32), 'l_M17': array([5.e-06],
# dtype=float32), 'nf_M17': array([10.], dtype=float32), 'IB': array([2.5e-05], dtype=float32)})])}
#
# env.observation_space_sample(): {'Agent_4': OrderedDict([('cur_param', OrderedDict([('IB', array([3.488859e-05],
# dtype=float32)), ('l_M11', array([3.7093325e-06], dtype=float32)), ('l_M12', array([5.409883e-06], dtype=float32)),
# ('l_M13', array([8.301396e-06], dtype=float32)), ('l_M14', array([8.997671e-06], dtype=float32)), ('l_M16',
# array([6.5396093e-06], dtype=float32)), ('l_M17', array([4.828313e-06], dtype=float32)), ('l_M18',
# array([4.5120787e-06], dtype=float32)), ('l_M19', array([5.567702e-06], dtype=float32)), ('l_M20',
# array([7.015079e-07], dtype=float32)), ('l_M21', array([7.994281e-06], dtype=float32)), ('l_M22',
# array([4.0017635e-06], dtype=float32)), ('l_M23', array([2.6731354e-06], dtype=float32)), ('l_M24',
# array([9.473896e-06], dtype=float32)), ('l_M25', array([3.6161491e-06], dtype=float32)), ('l_M35',
# array([2.8980012e-06], dtype=float32)), ('l_M36', array([3.202745e-06], dtype=float32)), ('nf_M11',
# array([5.7250543], dtype=float32)), ('nf_M12', array([5.7584395], dtype=float32)), ('nf_M13', array([19.127413],
# dtype=float32)), ('nf_M14', array([5.277461], dtype=float32)), ('nf_M16', array([7.623843], dtype=float32)),
# ('nf_M17', array([19.859924], dtype=float32)), ('nf_M18', array([1.3412921], dtype=float32)), ('nf_M19',
# array([19.165771], dtype=float32)), ('nf_M20', array([17.658527], dtype=float32)), ('nf_M21', array([16.550482],
# dtype=float32)), ('nf_M22', array([10.503215], dtype=float32)), ('nf_M23', array([15.773391], dtype=float32)),
# ('nf_M24', array([7.0365777], dtype=float32)), ('nf_M25', array([19.794416], dtype=float32)), ('nf_M35',
# array([8.182031], dtype=float32)), ('nf_M36', array([19.867239], dtype=float32)), ('w_M11_per_finger',
# array([6.253999e-06], dtype=float32)), ('w_M12_per_finger', array([9.312908e-06], dtype=float32)),
# ('w_M13_per_finger', array([5.4700818e-06], dtype=float32)), ('w_M14_per_finger', array([1.3282811e-06],
# dtype=float32)), ('w_M16_per_finger', array([9.624941e-06], dtype=float32)), ('w_M17_per_finger',
# array([7.978146e-06], dtype=float32)), ('w_M18_per_finger', array([6.2681825e-06], dtype=float32)),
# ('w_M19_per_finger', array([3.840373e-06], dtype=float32)), ('w_M20_per_finger', array([7.4867776e-06],
# dtype=float32)), ('w_M21_per_finger', array([4.9368095e-06], dtype=float32)), ('w_M22_per_finger',
# array([3.8444327e-06], dtype=float32)), ('w_M23_per_finger', array([5.240622e-06], dtype=float32)),
# ('w_M24_per_finger', array([5.739692e-06], dtype=float32)), ('w_M25_per_finger', array([4.865393e-06],
# dtype=float32)), ('w_M35_per_finger', array([9.400894e-06], dtype=float32)), ('w_M36_per_finger',
# array([8.428615e-06], dtype=float32))])), ('cur_specs', OrderedDict([('gainBandWidth', array([0.36878034],
# dtype=float32)), ('phaseMargin', array([0.7831989], dtype=float32)), ('powerSupplyRejectionRatio',
# array([0.2516507], dtype=float32)), ('pwr', array([0.94484717], dtype=float32)), ('slewRateDown',
# array([0.9619389], dtype=float32)), ('slewRateUp', array([0.10614631], dtype=float32))])), ('ideal_specs',
# OrderedDict([('gainBandWidth', array([0.91942436], dtype=float32)), ('phaseMargin', array([0.6696656],
# dtype=float32)), ('powerSupplyRejectionRatio', array([0.3096922], dtype=float32)), ('pwr', array([0.7791964],
# dtype=float32)), ('slewRateDown', array([0.5025928], dtype=float32)), ('slewRateUp', array([0.63289315],
# dtype=float32))]))]), 'Agent_2': OrderedDict([('cur_param', OrderedDict([('IB', array([1.2196229e-05],
# dtype=float32)), ('l_M11', array([2.0776538e-06], dtype=float32)), ('l_M12', array([9.9316785e-06],
# dtype=float32)), ('l_M13', array([1.2357418e-06], dtype=float32)), ('l_M14', array([7.899947e-06], dtype=float32)),
# ('l_M16', array([7.008056e-07], dtype=float32)), ('l_M17', array([4.782896e-06], dtype=float32)), ('l_M18',
# array([4.743843e-06], dtype=float32)), ('l_M19', array([5.7916986e-07], dtype=float32)), ('l_M20',
# array([4.900699e-06], dtype=float32)), ('l_M21', array([3.1617296e-06], dtype=float32)), ('l_M22',
# array([5.897941e-06], dtype=float32)), ('l_M23', array([6.8006866e-06], dtype=float32)), ('l_M24',
# array([3.4490483e-06], dtype=float32)), ('l_M25', array([5.457313e-06], dtype=float32)), ('l_M35',
# array([5.1510574e-06], dtype=float32)), ('l_M36', array([3.2094492e-06], dtype=float32)), ('nf_M11',
# array([5.5178423], dtype=float32)), ('nf_M12', array([17.096552], dtype=float32)), ('nf_M13', array([5.372219],
# dtype=float32)), ('nf_M14', array([9.30261], dtype=float32)), ('nf_M16', array([17.790192], dtype=float32)),
# ('nf_M17', array([19.361347], dtype=float32)), ('nf_M18', array([12.327857], dtype=float32)), ('nf_M19',
# array([2.186049], dtype=float32)), ('nf_M20', array([6.2998877], dtype=float32)), ('nf_M21', array([4.888271],
# dtype=float32)), ('nf_M22', array([2.7074447], dtype=float32)), ('nf_M23', array([12.15341], dtype=float32)),
# ('nf_M24', array([19.59236], dtype=float32)), ('nf_M25', array([3.7899017], dtype=float32)), ('nf_M35',
# array([4.06089], dtype=float32)), ('nf_M36', array([10.888589], dtype=float32)), ('w_M11_per_finger',
# array([4.8605825e-06], dtype=float32)), ('w_M12_per_finger', array([6.4940973e-06], dtype=float32)),
# ('w_M13_per_finger', array([3.8059718e-06], dtype=float32)), ('w_M14_per_finger', array([6.765076e-06],
# dtype=float32)), ('w_M16_per_finger', array([5.6510457e-06], dtype=float32)), ('w_M17_per_finger',
# array([2.6011553e-06], dtype=float32)), ('w_M18_per_finger', array([4.756043e-06], dtype=float32)),
# ('w_M19_per_finger', array([1.8286888e-06], dtype=float32)), ('w_M20_per_finger', array([3.905489e-06],
# dtype=float32)), ('w_M21_per_finger', array([5.999641e-06], dtype=float32)), ('w_M22_per_finger',
# array([8.4677295e-06], dtype=float32)), ('w_M23_per_finger', array([6.018726e-07], dtype=float32)),
# ('w_M24_per_finger', array([8.974966e-06], dtype=float32)), ('w_M25_per_finger', array([8.326707e-06],
# dtype=float32)), ('w_M35_per_finger', array([4.940441e-06], dtype=float32)), ('w_M36_per_finger',
# array([6.3070747e-06], dtype=float32))])), ('cur_specs', OrderedDict([('gainBandWidth', array([0.50004154],
# dtype=float32)), ('phaseMargin', array([0.8162017], dtype=float32)), ('powerSupplyRejectionRatio',
# array([0.8002812], dtype=float32)), ('pwr', array([0.31961608], dtype=float32)), ('slewRateDown',
# array([0.0625954], dtype=float32)), ('slewRateUp', array([0.70235187], dtype=float32))])), ('ideal_specs',
# OrderedDict([('gainBandWidth', array([0.3988291], dtype=float32)), ('phaseMargin', array([0.22294925],
# dtype=float32)), ('powerSupplyRejectionRatio', array([0.10329607], dtype=float32)), ('pwr', array([0.20135267],
# dtype=float32)), ('slewRateDown', array([0.9705978], dtype=float32)), ('slewRateUp', array([0.22718877],
# dtype=float32))]))]), 'Agent_1': OrderedDict([('cur_param', OrderedDict([('IB', array([4.8733593e-05],
# dtype=float32)), ('l_M11', array([9.89541e-06], dtype=float32)), ('l_M12', array([3.306946e-06], dtype=float32)),
# ('l_M13', array([9.807704e-06], dtype=float32)), ('l_M14', array([7.494183e-06], dtype=float32)), ('l_M16',
# array([4.875009e-06], dtype=float32)), ('l_M17', array([2.053644e-06], dtype=float32)), ('l_M18',
# array([9.263006e-06], dtype=float32)), ('l_M19', array([5.7012485e-06], dtype=float32)), ('l_M20',
# array([5.8266405e-06], dtype=float32)), ('l_M21', array([4.431855e-06], dtype=float32)), ('l_M22',
# array([1.4277853e-06], dtype=float32)), ('l_M23', array([6.5107474e-06], dtype=float32)), ('l_M24',
# array([6.981165e-06], dtype=float32)), ('l_M25', array([5.840813e-06], dtype=float32)), ('l_M35',
# array([2.7381598e-06], dtype=float32)), ('l_M36', array([7.4761924e-06], dtype=float32)), ('nf_M11',
# array([16.336792], dtype=float32)), ('nf_M12', array([4.2711735], dtype=float32)), ('nf_M13', array([2.8990934],
# dtype=float32)), ('nf_M14', array([7.034271], dtype=float32)), ('nf_M16', array([9.35911], dtype=float32)),
# ('nf_M17', array([13.959029], dtype=float32)), ('nf_M18', array([13.532006], dtype=float32)), ('nf_M19',
# array([12.251854], dtype=float32)), ('nf_M20', array([17.991444], dtype=float32)), ('nf_M21', array([2.2875557],
# dtype=float32)), ('nf_M22', array([16.142347], dtype=float32)), ('nf_M23', array([11.338005], dtype=float32)),
# ('nf_M24', array([7.69926], dtype=float32)), ('nf_M25', array([1.3231548], dtype=float32)), ('nf_M35',
# array([10.083244], dtype=float32)), ('nf_M36', array([15.662551], dtype=float32)), ('w_M11_per_finger',
# array([3.1470188e-06], dtype=float32)), ('w_M12_per_finger', array([1.6160881e-06], dtype=float32)),
# ('w_M13_per_finger', array([3.8313774e-06], dtype=float32)), ('w_M14_per_finger', array([2.5926038e-06],
# dtype=float32)), ('w_M16_per_finger', array([5.068371e-06], dtype=float32)), ('w_M17_per_finger',
# array([9.177756e-06], dtype=float32)), ('w_M18_per_finger', array([2.0921825e-06], dtype=float32)),
# ('w_M19_per_finger', array([6.1778346e-06], dtype=float32)), ('w_M20_per_finger', array([2.8211477e-06],
# dtype=float32)), ('w_M21_per_finger', array([1.713396e-06], dtype=float32)), ('w_M22_per_finger',
# array([6.4580245e-06], dtype=float32)), ('w_M23_per_finger', array([6.459355e-06], dtype=float32)),
# ('w_M24_per_finger', array([2.9485477e-06], dtype=float32)), ('w_M25_per_finger', array([8.667864e-06],
# dtype=float32)), ('w_M35_per_finger', array([9.1289274e-07], dtype=float32)), ('w_M36_per_finger',
# array([4.3968403e-06], dtype=float32))])), ('cur_specs', OrderedDict([('gainBandWidth', array([0.01800331],
# dtype=float32)), ('phaseMargin', array([0.17772032], dtype=float32)), ('powerSupplyRejectionRatio',
# array([0.11480644], dtype=float32)), ('pwr', array([0.14738902], dtype=float32)), ('slewRateDown',
# array([0.44913292], dtype=float32)), ('slewRateUp', array([0.28388909], dtype=float32))])), ('ideal_specs',
# OrderedDict([('gainBandWidth', array([0.50450426], dtype=float32)), ('phaseMargin', array([0.66819465],
# dtype=float32)), ('powerSupplyRejectionRatio', array([0.03752114], dtype=float32)), ('pwr', array([0.5920006],
# dtype=float32)), ('slewRateDown', array([0.73584205], dtype=float32)), ('slewRateUp', array([0.9266996],
# dtype=float32))]))]), 'Agent_3': OrderedDict([('cur_param', OrderedDict([('IB', array([3.7129394e-05],
# dtype=float32)), ('l_M11', array([6.0955936e-06], dtype=float32)), ('l_M12', array([8.8403e-06], dtype=float32)),
# ('l_M13', array([6.3082844e-06], dtype=float32)), ('l_M14', array([9.052247e-06], dtype=float32)), ('l_M16',
# array([8.837847e-06], dtype=float32)), ('l_M17', array([8.1810895e-06], dtype=float32)), ('l_M18',
# array([6.7831893e-06], dtype=float32)), ('l_M19', array([7.409512e-07], dtype=float32)), ('l_M20',
# array([9.156605e-07], dtype=float32)), ('l_M21', array([5.0708286e-06], dtype=float32)), ('l_M22',
# array([7.215578e-07], dtype=float32)), ('l_M23', array([5.2023906e-06], dtype=float32)), ('l_M24',
# array([5.701156e-06], dtype=float32)), ('l_M25', array([2.0911343e-06], dtype=float32)), ('l_M35',
# array([7.725545e-06], dtype=float32)), ('l_M36', array([4.704516e-06], dtype=float32)), ('nf_M11',
# array([7.4948115], dtype=float32)), ('nf_M12', array([16.665865], dtype=float32)), ('nf_M13', array([16.985886],
# dtype=float32)), ('nf_M14', array([10.597537], dtype=float32)), ('nf_M16', array([2.5796237], dtype=float32)),
# ('nf_M17', array([3.2329915], dtype=float32)), ('nf_M18', array([9.5885], dtype=float32)), ('nf_M19',
# array([15.947795], dtype=float32)), ('nf_M20', array([18.969822], dtype=float32)), ('nf_M21', array([15.1888],
# dtype=float32)), ('nf_M22', array([17.598837], dtype=float32)), ('nf_M23', array([3.1331208], dtype=float32)),
# ('nf_M24', array([11.780105], dtype=float32)), ('nf_M25', array([11.011333], dtype=float32)), ('nf_M35',
# array([14.618554], dtype=float32)), ('nf_M36', array([16.982437], dtype=float32)), ('w_M11_per_finger',
# array([4.3247583e-06], dtype=float32)), ('w_M12_per_finger', array([8.320992e-07], dtype=float32)),
# ('w_M13_per_finger', array([7.965043e-06], dtype=float32)), ('w_M14_per_finger', array([5.0882745e-06],
# dtype=float32)), ('w_M16_per_finger', array([7.973106e-06], dtype=float32)), ('w_M17_per_finger',
# array([7.748031e-06], dtype=float32)), ('w_M18_per_finger', array([7.083465e-06], dtype=float32)),
# ('w_M19_per_finger', array([1.0604557e-06], dtype=float32)), ('w_M20_per_finger', array([2.05271e-06],
# dtype=float32)), ('w_M21_per_finger', array([3.5939079e-06], dtype=float32)), ('w_M22_per_finger',
# array([8.666562e-06], dtype=float32)), ('w_M23_per_finger', array([1.3173719e-06], dtype=float32)),
# ('w_M24_per_finger', array([6.6037383e-06], dtype=float32)), ('w_M25_per_finger', array([9.990238e-06],
# dtype=float32)), ('w_M35_per_finger', array([7.610257e-06], dtype=float32)), ('w_M36_per_finger',
# array([9.586536e-06], dtype=float32))])), ('cur_specs', OrderedDict([('gainBandWidth', array([0.49476138],
# dtype=float32)), ('phaseMargin', array([0.7566147], dtype=float32)), ('powerSupplyRejectionRatio',
# array([0.4871909], dtype=float32)), ('pwr', array([0.13578035], dtype=float32)), ('slewRateDown',
# array([0.3167547], dtype=float32)), ('slewRateUp', array([0.24514116], dtype=float32))])), ('ideal_specs',
# OrderedDict([('gainBandWidth', array([0.46330574], dtype=float32)), ('phaseMargin', array([0.7833038],
# dtype=float32)), ('powerSupplyRejectionRatio', array([0.0299317], dtype=float32)), ('pwr', array([0.7358401],
# dtype=float32)), ('slewRateDown', array([0.7087019], dtype=float32)), ('slewRateUp', array([0.36469144],
# dtype=float32))]))])}
