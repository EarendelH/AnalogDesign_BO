import gymnasium
import numpy as np
import yaml

from util.util_func import unit_conversion
# from util_func import unit_conversion


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
    :param agent_assign_yaml_path: path of the agent assign yaml file
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
    # ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    #                       for key in sum(result_config.values(), [])}
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
# result_config_file = "../config_3/result.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
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

def gen_obs_space_extend(sim_config_file, param_range_config_file, agent_assign_yaml_path):
    """
    Generate observation space for the custom environment.
    :param agent_assign_yaml_path: path of the agent assign yaml file
    :param sim_config_file: path of the result config_3 file
    :param param_range_config_file: path of the parameter range config_3 file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Import YAML file
    with open(sim_config_file, 'r') as file:
        sim_config = yaml.safe_load(file)

    result_config = {}
    for item in sim_config:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        result_config[sim_name] = sim_item

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


# Test Code
# sim_config_file = "../config_3/simulation.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
# agent_assign_yaml_file = "../config_3/agent_assign.yaml"
# observation_space = gen_obs_space_extend(sim_config_file, param_range_config_file, agent_assign_yaml_file)
# print(f"observation_space: {observation_space}")


# Output observation_space:
# Dict('Agent_1': Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,), float32),
# 'l_M11': Box(5e-07, 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07, 1e-05,
# (1,), float32), 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32), 'l_M17': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05, (1,), float32),
# 'l_M20': Box(5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(5e-07, 1e-05,
# (1,), float32), 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32), 'l_M25': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05, (1,), float32),
# 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(1.0, 20.0, (1,),
# float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32), 'nf_M17': Box(1.0,
# 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,), float32),
# 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0, 20.0, (1,),
# float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32), 'nf_M25': Box(1.0,
# 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,), float32),
# 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,), float32)),
# 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32), 'slewRateDown':
# Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32)),
# 'ideal_specs': Dict('gainBandWidth':
# Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32), 'powerSupplyRejectionRatio': Box(-1.0,
# 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32), 'slewRateDown': Box(-1.0, 1.0, (1,), float32),
# 'slewRateUp': Box(-1.0, 1.0, (1,), float32))),
# 'Agent_2': Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,),
# float32), 'l_M11': Box(5e-07, 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07,
# 1e-05, (1,), float32), 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32),
# 'l_M17': Box(5e-07, 1e-05, (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05,
# (1,), float32), 'l_M20': Box(5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32),
# 'l_M25': Box(5e-07, 1e-05, (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05,
# (1,), float32), 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(
# 1.0, 20.0, (1,), float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32),
# 'nf_M17': Box(1.0, 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,),
# float32), 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0,
# 20.0, (1,), float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32),
# 'nf_M25': Box(1.0, 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,),
# float32), 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,),
# float32)), 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,),
# float32), 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32),
# 'slewRateDown': Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict(
# 'gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32), 'slewRateDown':
# Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32))),
# 'Agent_3': Dict('cur_param': Dict(
# 'IB': Box(1e-06, 5e-05, (1,), float32), 'l_M11': Box(5e-07, 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,),
# float32), 'l_M13': Box(5e-07, 1e-05, (1,), float32), 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07,
# 1e-05, (1,), float32), 'l_M17': Box(5e-07, 1e-05, (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32),
# 'l_M19': Box(5e-07, 1e-05, (1,), float32), 'l_M20': Box(5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05,
# (1,), float32), 'l_M22': Box(5e-07, 1e-05, (1,), float32), 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M25': Box(5e-07, 1e-05, (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32),
# 'l_M36': Box(5e-07, 1e-05, (1,), float32), 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,),
# float32), 'nf_M13': Box(1.0, 20.0, (1,), float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0,
# 20.0, (1,), float32), 'nf_M17': Box(1.0, 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32),
# 'nf_M19': Box(1.0, 20.0, (1,), float32), 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,),
# float32), 'nf_M22': Box(1.0, 20.0, (1,), float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0,
# 20.0, (1,), float32), 'nf_M25': Box(1.0, 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32),
# 'nf_M36': Box(1.0, 20.0, (1,), float32), 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger':
# Box(5e-07, 1e-05, (1,), float32), 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(
# 5e-07, 1e-05, (1,), float32), 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07,
# 1e-05, (1,), float32), 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05,
# (1,), float32), 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,),
# float32)), 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,),
# float32), 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32),
# 'slewRateDown': Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict(
# 'gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32), 'slewRateDown':
# Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32))),
# 'Agent_4': Dict('cur_param': Dict(
# 'IB': Box(1e-06, 5e-05, (1,), float32), 'l_M11': Box(5e-07, 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,),
# float32), 'l_M13': Box(5e-07, 1e-05, (1,), float32), 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07,
# 1e-05, (1,), float32), 'l_M17': Box(5e-07, 1e-05, (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32),
# 'l_M19': Box(5e-07, 1e-05, (1,), float32), 'l_M20': Box(5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05,
# (1,), float32), 'l_M22': Box(5e-07, 1e-05, (1,), float32), 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M25': Box(5e-07, 1e-05, (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32),
# 'l_M36': Box(5e-07, 1e-05, (1,), float32), 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,),
# float32), 'nf_M13': Box(1.0, 20.0, (1,), float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0,
# 20.0, (1,), float32), 'nf_M17': Box(1.0, 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32),
# 'nf_M19': Box(1.0, 20.0, (1,), float32), 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,),
# float32), 'nf_M22': Box(1.0, 20.0, (1,), float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0,
# 20.0, (1,), float32), 'nf_M25': Box(1.0, 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32),
# 'nf_M36': Box(1.0, 20.0, (1,), float32), 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger':
# Box(5e-07, 1e-05, (1,), float32), 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(
# 5e-07, 1e-05, (1,), float32), 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07,
# 1e-05, (1,), float32), 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05,
# (1,), float32), 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,),
# float32), 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,),
# float32)), 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,),
# float32), 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32),
# 'slewRateDown': Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict(
# 'gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'powerSupplyRejectionRatio': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32), 'slewRateDown':
# Box(-1.0, 1.0, (1,), float32), 'slewRateUp': Box(-1.0, 1.0, (1,), float32))))


def flatten_obs_space(obs_space: gymnasium.spaces.Dict):
    """
    :param obs_space: Complex obs dict
    :return: obs_space_flat: Flatten obs dict, only remain first level keys. For sub-dict, sort the element and convert
    to tuple. Rearrange the dict as 'cur_specs', 'ideal_specs' and 'cur_param'.
    """
    flattened_space = {}

    for agent, agent_space in obs_space.items():
        # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
        sorted_cur_specs = {key: agent_space['cur_specs'][key] for key in sorted(agent_space['cur_specs'])}
        sorted_ideal_specs = {key: agent_space['ideal_specs'][key] for key in sorted(agent_space['ideal_specs'])}
        sorted_cur_param = {key: agent_space['cur_param'][key] for key in sorted(agent_space['cur_param'])}

        # Print the sorted dicts for debugging
        # print(f"Agent: {agent}")
        # print("Sorted cur_specs:", sorted_cur_specs)
        # print("Sorted ideal_specs:", sorted_ideal_specs)
        # print("Sorted cur_param:", sorted_cur_param)

        # Combine all boxes from sorted dicts
        combined_boxes = []
        combined_boxes.extend(sorted_cur_specs.values())
        combined_boxes.extend(sorted_ideal_specs.values())
        combined_boxes.extend(sorted_cur_param.values())

        # Convert the combined list to a Tuple space and assign to the agent
        flattened_space[agent] = gymnasium.spaces.Tuple(combined_boxes)

    return gymnasium.spaces.Dict(flattened_space)


# Test Code
# result_config_file = "../config_3/result.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
# agent_assign_yaml_file = "../config_3/agent_assign.yaml"
# observation_space = gen_obs_space_extend(result_config_file, param_range_config_file, agent_assign_yaml_file)
# print(f"observation_space: {observation_space}")
# observation_space_flat = flatten_obs_space(observation_space)
# print(f"observation_space_flat: {observation_space_flat}")


def gen_obs_space_w_type(sim_config_file, param_range_config_file, agent_assign_yaml_path):
    """
    Generate observation space for the custom environment.
    :param agent_assign_yaml_path: path of the agent assign yaml file
    :param sim_config_file: path of the result config file
    :param param_range_config_file: path of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Import YAML file
    with open(sim_config_file, 'r') as file:
        sim_config = yaml.safe_load(file)

    result_config = {}
    for item in sim_config:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        result_config[sim_name] = sim_item

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

    # 1 for pch, 2 for nch, 3 for vsource, 4 for isource
    device_type_space = {}
    for component, data in param_range_config.items():
        if component == 'other_variable':
            for param in data['params']:
                variable_name = param['variable_name']
                device_type_space[variable_name] = gymnasium.spaces.Discrete(4)
        else:
            variable_name = component
            device_type_space[variable_name] = gymnasium.spaces.Discrete(4)

    # Combine three dicts into one gymnasium.spaces.Dict
    obs_space_single = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
        'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
        'cur_param': gymnasium.spaces.Dict(cur_param_spaces),
        'device_type': gymnasium.spaces.Dict(device_type_space)
    })

    obs_space = gymnasium.spaces.Dict()
    for group_name in agent_assign.keys():
        obs_space[group_name] = obs_space_single

    return obs_space


def flatten_obs_space_w_type(obs_space: gymnasium.spaces.Dict):
    """
    :param obs_space: Complex obs dict
    :return: obs_space_flat: Flatten obs dict, only remain first level keys. For sub-dict, sort the element and convert
    to tuple. Rearrange the dict as 'cur_specs', 'ideal_specs' and 'cur_param'.
    """
    flattened_space = {}

    for agent, agent_space in obs_space.items():
        # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
        sorted_cur_specs = {key: agent_space['cur_specs'][key] for key in sorted(agent_space['cur_specs'])}
        sorted_ideal_specs = {key: agent_space['ideal_specs'][key] for key in sorted(agent_space['ideal_specs'])}
        sorted_cur_param = {key: agent_space['cur_param'][key] for key in sorted(agent_space['cur_param'])}
        sorted_device_type = {key: agent_space['device_type'][key] for key in sorted(agent_space['device_type'])}

        # Print the sorted dicts for debugging
        # print(f"Agent: {agent}")
        # print("Sorted cur_specs:", sorted_cur_specs)
        # print("Sorted ideal_specs:", sorted_ideal_specs)
        # print("Sorted cur_param:", sorted_cur_param)

        # Combine all boxes from sorted dicts
        combined_boxes = []
        combined_boxes.extend(sorted_cur_specs.values())
        combined_boxes.extend(sorted_ideal_specs.values())
        combined_boxes.extend(sorted_cur_param.values())
        combined_boxes.extend(sorted_device_type.values())

        # Convert the combined list to a Tuple space and assign to the agent
        flattened_space[agent] = gymnasium.spaces.Tuple(combined_boxes)

    return gymnasium.spaces.Dict(flattened_space)


# Test Code
# result_config_file = "../config_3/simulation.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
# agent_assign_yaml_file = "../config_3/agent_assign.yaml"
# observation_space = gen_obs_space_w_type(result_config_file, param_range_config_file, agent_assign_yaml_file)
# print(f"observation_space: {observation_space}")
# observation_space_flat = flatten_obs_space_w_type(observation_space)
# print(f"observation_space_flat: {observation_space_flat}")


def gen_obs_space_w_region(sim_config_file, param_range_config_file, agent_assign_yaml_path):
    """
    Generate observation space for the custom environment.
    :param agent_assign_yaml_path: path of the agent assign yaml file
    :param sim_config_file: path of the result config file
    :param param_range_config_file: path of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Import YAML file
    with open(sim_config_file, 'r') as file:
        sim_config = yaml.safe_load(file)

    result_config = {}
    for item in sim_config:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        result_config[sim_name] = sim_item

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

    # Create spaces for transistor region
    transistor_region_space = {}
    for component, data in param_range_config.items():
        if component == 'other_variable':
            pass
        else:
            variable_name = component
            # 0 cut-off, 1 triode, 2 saturation, 3 sub-th, 4 breakdown
            transistor_region_space[variable_name] = gymnasium.spaces.Discrete(5)

    # Combine three dicts into one gymnasium.spaces.Dict
    obs_space_single = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
        'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
        'cur_param': gymnasium.spaces.Dict(cur_param_spaces),
        'transistor_region': gymnasium.spaces.Dict(transistor_region_space)
    })

    obs_space = gymnasium.spaces.Dict()
    for group_name in agent_assign.keys():
        obs_space[group_name] = obs_space_single

    return obs_space


def flatten_obs_space_w_region(obs_space: gymnasium.spaces.Dict):
    """
    :param obs_space: Complex obs dict
    :return: obs_space_flat: Flatten obs dict, only remain first level keys. For sub-dict, sort the element and convert
    to tuple. Rearrange the dict as 'cur_specs', 'ideal_specs' and 'cur_param'.
    """
    flattened_space = {}

    for agent, agent_space in obs_space.items():
        # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
        sorted_cur_specs = {key: agent_space['cur_specs'][key] for key in sorted(agent_space['cur_specs'])}
        sorted_ideal_specs = {key: agent_space['ideal_specs'][key] for key in sorted(agent_space['ideal_specs'])}
        sorted_cur_param = {key: agent_space['cur_param'][key] for key in sorted(agent_space['cur_param'])}
        sorted_transistor_region = {key: agent_space['transistor_region'][key] for key in
                                    sorted(agent_space['transistor_region'])}

        # Print the sorted dicts for debugging
        # print(f"Agent: {agent}")
        # print("Sorted cur_specs:", sorted_cur_specs)
        # print("Sorted ideal_specs:", sorted_ideal_specs)
        # print("Sorted cur_param:", sorted_cur_param)

        # Combine all boxes from sorted dicts
        combined_boxes = []
        combined_boxes.extend(sorted_cur_specs.values())
        combined_boxes.extend(sorted_ideal_specs.values())
        combined_boxes.extend(sorted_cur_param.values())
        combined_boxes.extend(sorted_transistor_region.values())

        # Convert the combined list to a Tuple space and assign to the agent
        flattened_space[agent] = gymnasium.spaces.Tuple(combined_boxes)

    return gymnasium.spaces.Dict(flattened_space)


# Test Code
# result_config_file = "../config_3/simulation.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
# agent_assign_yaml_file = "../config_3/agent_assign.yaml"
# observation_space = gen_obs_space_w_region(result_config_file, param_range_config_file, agent_assign_yaml_file)
# print(f"observation_space: {observation_space}")
# observation_space_flat = flatten_obs_space_w_region(observation_space)
# print(f"observation_space_flat: {observation_space_flat}")

# Output
# observation_space: Dict('Agent_1': Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,), float32),
# 'l_M11': Box(5e-07, 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07, 1e-05,
# (1,), float32), 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32), 'l_M17': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05, (1,), float32),
# 'l_M20': Box(5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(5e-07, 1e-05,
# (1,), float32), 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32), 'l_M25': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05, (1,), float32),
# 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(1.0, 20.0, (1,),
# float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32), 'nf_M17': Box(1.0,
# 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,), float32),
# 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0, 20.0, (1,),
# float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32), 'nf_M25': Box(1.0,
# 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,), float32),
# 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,), float32)),
# 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'pwr': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32),
# 'phaseMargin': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32)), 'transistor_region': Dict(
# 'M11': Discrete(5), 'M12': Discrete(5), 'M13': Discrete(5), 'M14': Discrete(5), 'M16': Discrete(5),
# 'M17': Discrete(5), 'M18': Discrete(5), 'M19': Discrete(5), 'M20': Discrete(5), 'M21': Discrete(5),
# 'M22': Discrete(5), 'M23': Discrete(5), 'M24': Discrete(5), 'M25': Discrete(5), 'M35': Discrete(5),
# 'M36': Discrete(5))), 'Agent_2': Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,), float32), 'l_M11': Box(5e-07,
# 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07, 1e-05, (1,), float32),
# 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32), 'l_M17': Box(5e-07, 1e-05,
# (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05, (1,), float32), 'l_M20': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(5e-07, 1e-05, (1,), float32),
# 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32), 'l_M25': Box(5e-07, 1e-05,
# (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05, (1,), float32),
# 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(1.0, 20.0, (1,),
# float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32), 'nf_M17': Box(1.0,
# 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,), float32),
# 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0, 20.0, (1,),
# float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32), 'nf_M25': Box(1.0,
# 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,), float32),
# 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,), float32)),
# 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'pwr': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32),
# 'phaseMargin': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32)), 'transistor_region': Dict(
# 'M11': Discrete(5), 'M12': Discrete(5), 'M13': Discrete(5), 'M14': Discrete(5), 'M16': Discrete(5),
# 'M17': Discrete(5), 'M18': Discrete(5), 'M19': Discrete(5), 'M20': Discrete(5), 'M21': Discrete(5),
# 'M22': Discrete(5), 'M23': Discrete(5), 'M24': Discrete(5), 'M25': Discrete(5), 'M35': Discrete(5),
# 'M36': Discrete(5))), 'Agent_3': Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,), float32), 'l_M11': Box(5e-07,
# 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07, 1e-05, (1,), float32),
# 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32), 'l_M17': Box(5e-07, 1e-05,
# (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05, (1,), float32), 'l_M20': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(5e-07, 1e-05, (1,), float32),
# 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32), 'l_M25': Box(5e-07, 1e-05,
# (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05, (1,), float32),
# 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(1.0, 20.0, (1,),
# float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32), 'nf_M17': Box(1.0,
# 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,), float32),
# 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0, 20.0, (1,),
# float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32), 'nf_M25': Box(1.0,
# 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,), float32),
# 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,), float32)),
# 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'pwr': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32),
# 'phaseMargin': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32)), 'transistor_region': Dict(
# 'M11': Discrete(5), 'M12': Discrete(5), 'M13': Discrete(5), 'M14': Discrete(5), 'M16': Discrete(5),
# 'M17': Discrete(5), 'M18': Discrete(5), 'M19': Discrete(5), 'M20': Discrete(5), 'M21': Discrete(5),
# 'M22': Discrete(5), 'M23': Discrete(5), 'M24': Discrete(5), 'M25': Discrete(5), 'M35': Discrete(5),
# 'M36': Discrete(5))), 'Agent_4': Dict('cur_param': Dict('IB': Box(1e-06, 5e-05, (1,), float32), 'l_M11': Box(5e-07,
# 1e-05, (1,), float32), 'l_M12': Box(5e-07, 1e-05, (1,), float32), 'l_M13': Box(5e-07, 1e-05, (1,), float32),
# 'l_M14': Box(5e-07, 1e-05, (1,), float32), 'l_M16': Box(5e-07, 1e-05, (1,), float32), 'l_M17': Box(5e-07, 1e-05,
# (1,), float32), 'l_M18': Box(5e-07, 1e-05, (1,), float32), 'l_M19': Box(5e-07, 1e-05, (1,), float32), 'l_M20': Box(
# 5e-07, 1e-05, (1,), float32), 'l_M21': Box(5e-07, 1e-05, (1,), float32), 'l_M22': Box(5e-07, 1e-05, (1,), float32),
# 'l_M23': Box(5e-07, 1e-05, (1,), float32), 'l_M24': Box(5e-07, 1e-05, (1,), float32), 'l_M25': Box(5e-07, 1e-05,
# (1,), float32), 'l_M35': Box(5e-07, 1e-05, (1,), float32), 'l_M36': Box(5e-07, 1e-05, (1,), float32),
# 'nf_M11': Box(1.0, 20.0, (1,), float32), 'nf_M12': Box(1.0, 20.0, (1,), float32), 'nf_M13': Box(1.0, 20.0, (1,),
# float32), 'nf_M14': Box(1.0, 20.0, (1,), float32), 'nf_M16': Box(1.0, 20.0, (1,), float32), 'nf_M17': Box(1.0,
# 20.0, (1,), float32), 'nf_M18': Box(1.0, 20.0, (1,), float32), 'nf_M19': Box(1.0, 20.0, (1,), float32),
# 'nf_M20': Box(1.0, 20.0, (1,), float32), 'nf_M21': Box(1.0, 20.0, (1,), float32), 'nf_M22': Box(1.0, 20.0, (1,),
# float32), 'nf_M23': Box(1.0, 20.0, (1,), float32), 'nf_M24': Box(1.0, 20.0, (1,), float32), 'nf_M25': Box(1.0,
# 20.0, (1,), float32), 'nf_M35': Box(1.0, 20.0, (1,), float32), 'nf_M36': Box(1.0, 20.0, (1,), float32),
# 'w_M11_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M12_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M13_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M14_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M16_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M17_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M18_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M19_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M20_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M21_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M22_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M23_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M24_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M25_per_finger': Box(5e-07, 1e-05, (1,), float32),
# 'w_M35_per_finger': Box(5e-07, 1e-05, (1,), float32), 'w_M36_per_finger': Box(5e-07, 1e-05, (1,), float32)),
# 'cur_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32), 'phaseMargin': Box(-1.0, 1.0, (1,), float32),
# 'pwr': Box(-1.0, 1.0, (1,), float32)), 'ideal_specs': Dict('gainBandWidth': Box(-1.0, 1.0, (1,), float32),
# 'phaseMargin': Box(-1.0, 1.0, (1,), float32), 'pwr': Box(-1.0, 1.0, (1,), float32)), 'transistor_region': Dict(
# 'M11': Discrete(5), 'M12': Discrete(5), 'M13': Discrete(5), 'M14': Discrete(5), 'M16': Discrete(5),
# 'M17': Discrete(5), 'M18': Discrete(5), 'M19': Discrete(5), 'M20': Discrete(5), 'M21': Discrete(5),
# 'M22': Discrete(5), 'M23': Discrete(5), 'M24': Discrete(5), 'M25': Discrete(5), 'M35': Discrete(5),
# 'M36': Discrete(5))))

# observation_space_flat: Dict('Agent_1': Tuple(Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32),
# Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,),
# float32), Box(1e-06, 5e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32),
# Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,),
# float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0,
# 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32),
# Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05,
# (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5)), 'Agent_2': Tuple(Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,),
# float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0,
# 1.0, (1,), float32), Box(1e-06, 5e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,
# ), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0,
# 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32),
# Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,),
# float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Discrete(5), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5)), 'Agent_3': Tuple(Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0,
# (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32),
# Box(-1.0, 1.0, (1,), float32), Box(1e-06, 5e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0,
# 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32),
# Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,),
# float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0,
# 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5)), 'Agent_4': Tuple(Box(-1.0, 1.0, (1,), float32),
# Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,), float32), Box(-1.0, 1.0, (1,),
# float32), Box(-1.0, 1.0, (1,), float32), Box(1e-06, 5e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,),
# float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0,
# 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32),
# Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,),
# float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(1.0, 20.0, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32),
# Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07,
# 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,),
# float32), Box(5e-07, 1e-05, (1,), float32), Box(5e-07, 1e-05, (1,), float32), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5),
# Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5), Discrete(5)))
