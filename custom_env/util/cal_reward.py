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

    # epsilon = 0.1

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

        rew += float(single_reward)

    if rew >= 0:
        rew = rew + 15
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

            rew += float(single_reward)*2  # 2 is the weight for the optimal reward

    return rew


# Test Code

# ideal_specs = {'DC_IQ': {'objective': 'min', 'reward_type': 'optimal', 'value': 8e-06},
# 'Stability_1m_gainBandWidth': {'objective': 'max', 'reward_type': 'optimal', 'value': 1000000.0},
# 'Stability_1m_phaseMargin': {'objective': 'max', 'reward_type': 'satisfactory', 'value': 30.0},
# 'Stability_50m_gainBandWidth': {'objective': 'max', 'reward_type': 'optimal', 'value': 1000000.0},
# 'Stability_50m_phaseMargin': {'objective': 'max', 'reward_type': 'satisfactory', 'value': 30.0},
# 'Stability_100m_gainBandWidth': {'objective': 'max', 'reward_type': 'optimal', 'value': 1000000.0},
# 'Stability_100m_phaseMargin': {'objective': 'max', 'reward_type': 'satisfactory', 'value': 30.0},
# 'Load_Reg_loadReg': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.12}, 'Line_Reg_100m_lineReg': {
# 'objective': 'min', 'reward_type': 'optimal', 'value': 0.008}, 'Line_Reg_1m_lineReg': {'objective': 'min',
# 'reward_type': 'optimal', 'value': 0.004}, 'Trans_1_2V_overShoot': {'objective': 'min', 'reward_type': 'optimal',
# 'value': 0.07}, 'Trans_1_2V_underShoot': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.06},
# 'Trans_0_75V_overShoot': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.11}, 'Trans_0_75V_underShoot': {
# 'objective': 'min', 'reward_type': 'optimal', 'value': 0.08}, 'Trans_Line_Reg_overShoot': {'objective': 'min',
# 'reward_type': 'optimal', 'value': 0.04}, 'Trans_Line_Reg_underShoot': {'objective': 'min', 'reward_type':
# 'optimal', 'value': 0.03}, 'PSR_psr_100': {'objective': 'max', 'reward_type': 'optimal', 'value': 50.0},
# 'PSR_psr_1k': {'objective': 'max', 'reward_type': 'optimal', 'value': 45.0}, 'PSR_psr_10k': {'objective': 'max',
# 'reward_type': 'optimal', 'value': 25.0}, 'PSR_psr_100k': {'objective': 'max', 'reward_type': 'optimal',
# 'value': 5.0}, 'PSR_psr_1M': {'objective': 'max', 'reward_type': 'optimal', 'value': 0.0001}}
#
# cur_specs = {'DC': {'DC_IQ': 0.000100364}, 'Stability_1m': {'Stability_1m_phaseMargin': 26.19896,
# 'Stability_1m_gainBandWidth': 54918080.0}, 'Stability_50m': {'Stability_50m_phaseMargin': 67.73379,
# 'Stability_50m_gainBandWidth': 93061670.0}, 'Stability_100m': {'Stability_100m_phaseMargin': 71.66593,
# 'Stability_100m_gainBandWidth': 89405280.0}, 'Load_Reg': {'Load_Reg_loadReg': 0.0041199999999996795},
# 'Line_Reg_100m': {'Line_Reg_100m_lineReg': 0.00017333333333338335}, 'Line_Reg_1m': {'Line_Reg_1m_lineReg':
# 5.333333333326332e-05}, 'Trans_1_2V': {'Trans_1_2V_overShoot': 0.13838300000000003, 'Trans_1_2V_underShoot':
# 0.13162599999999997}, 'Trans_0_75V': {'Trans_0_75V_overShoot': 0.11439599999999994, 'Trans_0_75V_underShoot':
# 0.12124299999999999}, 'Trans_Line_Reg': {'Trans_Line_Reg_overShoot': 0.005529999999999979,
# 'Trans_Line_Reg_underShoot': 0.005035000000000012}, 'PSR': {'PSR_psr_100': 83.75430209719951, 'PSR_psr_1k':
# 62.264008781888755, 'PSR_psr_10k': 44.26852949793264, 'PSR_psr_100k': 22.89905881726738, 'PSR_psr_1M':
# 5.665768656927508}}
#
# norm_specs = {'Stability_1m': {'Stability_1m_gainBandWidth': 1000000.0, 'Stability_1m_phaseMargin': 30.0},
# 'Stability_50m': {'Stability_50m_gainBandWidth': 1000000.0, 'Stability_50m_phaseMargin': 30.0}, 'Stability_100m': {
# 'Stability_100m_gainBandWidth': 1000000.0, 'Stability_100m_phaseMargin': 30.0}, 'Load_Reg': {'Load_Reg_loadReg':
# 0.12}, 'Line_Reg_100m': {'Line_Reg_100m_lineReg': 0.008}, 'Line_Reg_1m': {'Line_Reg_1m_lineReg': 0.004},
# 'Trans_1_2V': {'Trans_1_2V_overShoot': 0.07, 'Trans_1_2V_underShoot': 0.06}, 'Trans_0_75V': {
# 'Trans_0_75V_overShoot': 0.11, 'Trans_0_75V_underShoot': 0.08}, 'Trans_Line_Reg': {'Trans_Line_Reg_overShoot':
# 0.04, 'Trans_Line_Reg_underShoot': 0.03}, 'PSR': {'PSR_psr_100': 50.0, 'PSR_psr_1k': 45.0, 'PSR_psr_10k': 25.0,
# 'PSR_psr_100k': 5.0, 'PSR_psr_1M': 0.0001}, 'DC': {'DC_IQ': 8e-06}}
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
