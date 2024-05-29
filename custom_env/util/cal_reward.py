import yaml


def cal_reward(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    :param ideal_specs_dict: Dict with ideal specs value and property
    :param cur_specs_dict: Dict with current specs
    :param norm_specs_dict: Dict with normalized specs
    :return: reward: float, reward value
    """

    # Flatten cur_specs_dict
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}
    norm_specs_flatten = {k: v for d in norm_specs_dict.values() for k, v in d.items()}
    rew = 0
    min_rew = 0

    # Get the item number of cur_specs_flatten
    for spec, detail in ideal_specs_dict.items():

        if spec.startswith('DC'):
            min_rew_single = -10
        elif spec.startswith('Trans'):
            min_rew_single = -5
        else:
            min_rew_single = -1

        min_rew += min_rew_single

    for spec, detail in ideal_specs_dict.items():

        single_reward = 0

        ideal_spec_value = float(detail['value'])
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_objective = detail['objective']

        if constrain_objective == "max":
            single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
        elif constrain_objective == "min":
            single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
        if spec.startswith('DC'):
            single_reward = single_reward * 10
        elif spec.startswith('Trans'):
            single_reward = single_reward * 5

        rew += float(single_reward)

    rew = - rew / min_rew

    if rew >= 0:
        rew = rew + 10
        for spec, detail in ideal_specs_dict.items():

            single_reward = 0
            general_ideal_spec_value = float(norm_specs_flatten[spec])
            cur_spec_value = float(cur_specs_flatten[spec])
            reward_type = detail['reward_type']
            constrain_objective = detail['objective']

            if reward_type == "optimal":
                if constrain_objective == "max":
                    single_reward = max((cur_spec_value - general_ideal_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
                elif constrain_objective == "min":
                    single_reward = max((general_ideal_spec_value - cur_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
                if spec.startswith('DC'):
                    single_reward = single_reward * 10
                elif spec.startswith('Trans'):
                    single_reward = single_reward * 5

            rew += float(single_reward)

    return rew


# Test Code

# ideal_specs = {'DC_IQ': {'objective': 'min', 'reward_type': 'optimal', 'value': 1},
# 'Stability_1m_gainBandWidth': {'objective': 'max', 'reward_type': 'optimal', 'value': 1},
# 'Line_Reg_1m_lineReg': {'objective': 'min',
# 'reward_type': 'optimal', 'value': 1}, 'Trans_1_2V_overShoot': {'objective': 'min', 'reward_type': 'optimal',
# 'value': 1}, 'PSR_psr_100': {'objective': 'max', 'reward_type': 'optimal', 'value': 1}}
#
# cur_specs = {'DC': {'DC_IQ': 1}, 'Stability_1m': {'Stability_1m_gainBandWidth': 1},
# 'Line_Reg_1m': {'Line_Reg_1m_lineReg': 1}, 'Trans_1_2V': {'Trans_1_2V_overShoot': 1}, 'PSR':
# {'PSR_psr_100': 1}}
#
# norm_specs = {'DC': {'DC_IQ': 2}, 'Stability_1m': {'Stability_1m_gainBandWidth': 2},
# 'Line_Reg_1m': {'Line_Reg_1m_lineReg': 2}, 'Trans_1_2V': {'Trans_1_2V_overShoot': 2}, 'PSR':
# {'PSR_psr_100': 2}}
#
# reward = cal_reward(ideal_specs, cur_specs, norm_specs)
#
# print(reward)

# Output


def cal_reward_simple(ideal_specs_dict, cur_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    :param ideal_specs_dict: Dict with ideal specs value and property
    :param cur_specs_dict: Dict with current specs
    :return: reward: float, reward value
    """

    # Flatten cur_specs_dict
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}

    rew = 0

    epsilon = 0.1

    for spec, detail in ideal_specs_dict.items():

        single_reward = 0

        ideal_spec_value = float(detail['value'])
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_type = detail['constrain_type']
        constrain_objective = detail['objective']

        if constrain_type == 'hard' and constrain_objective == "max":
            single_reward = 0 if cur_spec_value >= ideal_spec_value else -1
        elif constrain_type == 'hard' and constrain_objective == "min":
            single_reward = 0 if cur_spec_value <= ideal_spec_value else -1
        elif constrain_type == 'soft' and constrain_objective == "min":
            single_reward = epsilon * (ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value)
        elif constrain_type == 'soft' and constrain_objective == "max":
            single_reward = epsilon * (cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value)
        # print(f"Debug!!! {spec}: {single_reward} ideal_spec_value: {ideal_spec_value} cur_spec_value: {
        # cur_spec_value}")
        rew += float(single_reward)

    rew = 5 if rew >= -0.01 else rew

    return rew
