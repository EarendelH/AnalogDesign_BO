import logging
import yaml

def cal_reward_buck(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {}
    for spec, detail in ideal_specs_dict.items():
        reward_weight[spec] = 1
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)


# Test Code

# ideal_specs = {'Efficiency_efficiency': {'objective': 'max', 'reward_type': 'optimal', 'value': 0.92},
#                'Efficiency_vo_mean': {'objective': 'max', 'reward_type': 'satisfactory', 'value': 0.927},
#                'Efficiency_vo_var': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.0000011}}
#
# cur_specs = {'Efficiency': {'Efficiency_efficiency': 0.95, 'Efficiency_vo_mean': 0.94, 'Efficiency_vo_var': 0.000001}}
#
# norm_specs = {
#     'Efficiency': {'Efficiency_efficiency': 0.92, 'Efficiency_vo_mean': 0.927, 'Efficiency_vo_var': 0.0000011}}
#
# reward = cal_reward_buck(ideal_specs, cur_specs, norm_specs)
# print(reward)

def cal_reward_LDO(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
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
    min_rew_bound = -5
    reward_weight = {}
    for spec, detail in ideal_specs_dict.items():
        reward_weight[spec] = 1
        if spec.startswith('DC'):
            reward_weight[spec] = 10
        elif spec.startswith('Trans'):
            reward_weight[spec] = 10
        elif spec.startswith('Load_Reg'):
            reward_weight[spec] = 5
    reward_weight_sum = sum(reward_weight.values())

    rew = 0

    for spec, detail in ideal_specs_dict.items():

        single_reward = 0
        ideal_spec_value = float(detail['value'])
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_objective = detail['objective']

        if constrain_objective == "max":
            single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
        elif constrain_objective == "min":
            single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)

        single_reward = single_reward * reward_weight[spec]
        rew += float(single_reward)
        # print(f"Debug, spec: {spec}, single_reward: {single_reward}, ideal_spec_value: {ideal_spec_value}, "
        #       f"cur_spec_value: {cur_spec_value}, constrain_objective: {constrain_objective}, "
        #       f"reward_weight: {reward_weight[spec]}")
    rew = -1 * rew / reward_weight_sum * min_rew_bound
    # print(f"Debug, rew: {rew}, reward_weight_sum: {reward_weight_sum}, min_rew_bound: {min_rew_bound}")

    if rew >= 0:
        rew = 10
        for spec, detail in ideal_specs_dict.items():
            single_reward = 0
            general_ideal_spec_value = float(norm_specs_flatten[spec])
            cur_spec_value = float(cur_specs_flatten[spec])
            reward_type = detail['reward_type']
            constrain_objective = detail['objective']

            if reward_type == "optimal":
                if constrain_objective == "max":
                    single_reward = max(
                        (cur_spec_value - general_ideal_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
                elif constrain_objective == "min":
                    single_reward = max(
                        (general_ideal_spec_value - cur_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)

            single_reward = single_reward * reward_weight[spec]
            rew += float(single_reward)
            # print(f"Debug, spec: {spec}, single_reward: {single_reward}, general_ideal_spec_value: "
            #       f"{general_ideal_spec_value}, cur_spec_value: {cur_spec_value}, constrain_objective: "
            #       f"{constrain_objective}, reward_type: {reward_type}, reward_weight: {reward_weight[spec]}")

    return rew


# Test Code

# ideal_specs = {'Efficiency_efficiency': {'objective': 'max', 'reward_type': 'optimal', 'value': 0.92},
#                'Efficiency_vo_mean': {'objective': 'max', 'reward_type': 'satisfactory', 'value': 0.927},
#                'Efficiency_vo_var': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.0000011},
#                'Load_Reg_loadReg': {'objective': 'min', 'reward_type': 'optimal', 'value': 1.2},
#                'Trans_overShoot': {'objective': 'min', 'reward_type': 'optimal', 'value': 1.3},
#                'DC_IQ': {'objective': 'min', 'reward_type': 'optimal', 'value': 1.4}}
#
# cur_specs = {'Efficiency': {'Efficiency_efficiency': 0.99, 'Efficiency_vo_mean': 0.99, 'Efficiency_vo_var': 0.000001},
#              'Load_Reg': {'Load_Reg_loadReg': 1.1}, 'Trans': {'Trans_overShoot': 1.1}, 'DC': {'DC_IQ': 1.1}}
#
# norm_specs = {'Efficiency': {'Efficiency_efficiency': 0.93, 'Efficiency_vo_mean': 1.1, 'Efficiency_vo_var': 0.000001},
#               'Load_Reg': {'Load_Reg_loadReg': 1}, 'Trans': {'Trans_overShoot': 1}, 'DC': {'DC_IQ': 1}}
#
# reward = cal_reward_LDO(ideal_specs, cur_specs, norm_specs)
# print(reward)


def cal_reward_Haoqiang(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.052951389,
        'Load_Reg_loadReg': 0.016493056,
        'Stability_FFRC_200u_phaseMargin': 0.016493056,
        'Stability_FFRC_200u_gainBandWidth': 0.016493056,
        'Stability_FFRC_25m_phaseMargin': 0.016493056,
        'Stability_FFRC_25m_gainBandWidth': 0.016493056,
        'Stability_FFRC_50m_phaseMargin': 0.016493056,
        'Stability_FFRC_50m_gainBandWidth': 0.016493056,
        'Stability_Main_200u_phaseMargin': 0.058823529,
        'Stability_Main_200u_gainBandWidth': 0.024305556,
        'Stability_Main_25m_phaseMargin': 0.016493056,
        'Stability_Main_25m_gainBandWidth': 0.021701389,
        'Stability_Main_50m_phaseMargin': 0.016493056,
        'Stability_Main_50m_gainBandWidth': 0.032986111,
        'Stability_Vset_phaseMargin': 0.016493056,
        'Stability_Vset_gainBandWidth': 0.016493056,
        'Trans_overShoot': 0.060763889,
        'Trans_underShoot': 0.119791667,
        'PSR_psr_10': 0.059027778,
        'PSR_psr_1k': 0.059895833,
        'PSR_psr_1M': 0.114583333,
        'PSR_psr_10M': 0.165798611,
        'PSR_psr_100M': 0.051215278
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_Haoqiang_psr(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.03410175,
        'Load_Reg_loadReg': 0.01062186,
        'Stability_FFRC_200u_phaseMargin': 0.01062186,
        'Stability_FFRC_200u_gainBandWidth': 0.01062186,
        'Stability_FFRC_25m_phaseMargin': 0.01062186,
        'Stability_FFRC_25m_gainBandWidth': 0.01062186,
        'Stability_FFRC_50m_phaseMargin': 0.01062186,
        'Stability_FFRC_50m_gainBandWidth': 0.01062186,
        'Stability_Main_200u_phaseMargin': 0.03788352,
        'Stability_Main_200u_gainBandWidth': 0.01565326,
        'Stability_Main_25m_phaseMargin': 0.01062186,
        'Stability_Main_25m_gainBandWidth': 0.01397613,
        'Stability_Main_50m_phaseMargin': 0.01062186,
        'Stability_Main_50m_gainBandWidth': 0.02124371,
        'Stability_Vset_phaseMargin': 0.01062186,
        'Stability_Vset_gainBandWidth': 0.01062186,
        'Trans_overShoot': 0.03913315,
        'Trans_underShoot': 0.07714821,
        'PSR_psr_10': 0.12880397,
        'PSR_psr_1k': 0.12880397,
        'PSR_psr_1M': 0.12880397,
        'PSR_psr_10M': 0.12880397,
        'PSR_psr_100M': 0.12880397
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_Haoqiang_Equal(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.0434783,
        'Load_Reg_loadReg': 0.0434783,
        'Stability_FFRC_200u_phaseMargin': 0.0434783,
        'Stability_FFRC_200u_gainBandWidth': 0.0434783,
        'Stability_FFRC_25m_phaseMargin': 0.0434783,
        'Stability_FFRC_25m_gainBandWidth': 0.0434783,
        'Stability_FFRC_50m_phaseMargin': 0.0434783,
        'Stability_FFRC_50m_gainBandWidth': 0.0434783,
        'Stability_Main_200u_phaseMargin': 0.0434783,
        'Stability_Main_200u_gainBandWidth': 0.0434783,
        'Stability_Main_25m_phaseMargin': 0.0434783,
        'Stability_Main_25m_gainBandWidth': 0.0434783,
        'Stability_Main_50m_phaseMargin': 0.0434783,
        'Stability_Main_50m_gainBandWidth': 0.0434783,
        'Stability_Vset_phaseMargin': 0.0434783,
        'Stability_Vset_gainBandWidth': 0.0434783,
        'Trans_overShoot': 0.0434783,
        'Trans_underShoot': 0.0434783,
        'PSR_psr_10': 0.0434783,
        'PSR_psr_1k': 0.0434783,
        'PSR_psr_1M': 0.0434783,
        'PSR_psr_10M': 0.0434783,
        'PSR_psr_100M': 0.0434783,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)


def cal_reward_Jianping(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.156195123,
        'Line_Reg_1m_lineReg': 0.01479131,
        'Line_Reg_100m_lineReg': 0.01479131,
        'Load_Reg_loadReg': 0.01479131,
        'PSR_100m_psr_100': 0.019973054,
        'PSR_100m_psr_1k': 0.021153642,
        'PSR_100m_psr_10k': 0.024854711,
        'PSR_100m_psr_100k': 0.038199924,
        'PSR_100m_psr_1M': 0.05220891,
        'PSR_1m_psr_100': 0.019973054,
        'PSR_1m_psr_1k': 0.021153642,
        'PSR_1m_psr_10k': 0.024854711,
        'PSR_1m_psr_100k': 0.038199924,
        'PSR_1m_psr_1M': 0.05220891,
        'Stability_1m_0_75V_gainBandWidth': 0.012809873,
        'Stability_1m_0_75V_phaseMargin': 0.059289968,
        'Stability_1m_1_2V_gainBandWidth': 0.012809873,
        'Stability_1m_1_2V_phaseMargin': 0.059289968,
        'Stability_100m_0_75V_gainBandWidth': 0.007474999,
        'Stability_100m_0_75V_phaseMargin': 0.008629036,
        'Stability_100m_1_2V_gainBandWidth': 0.007474999,
        'Stability_100m_1_2V_phaseMargin': 0.008629036,
        'Trans_0_75V_overShoot': 0.0712922,
        'Trans_0_75V_underShoot': 0.083829156,
        'Trans_1_2V_overShoot': 0.0712922,
        'Trans_1_2V_underShoot': 0.083829156
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_AXS(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.058823529,
        'Mismatch_sigma': 0.058823529,
        'Line_Reg_lineReg': 0.058823529,
        'Line_Reg_Trans_deltaVout': 0.058823529,
        'Load_Reg_loadReg': 0.058823529,
        'Load_Reg_Trans_startupShoot': 0.058823529,
        'Load_Reg_Trans_overShoot': 0.058823529,
        'Load_Reg_Trans_underShoot': 0.058823529,
        'PSR_psr_100k': 0.058823529,
        'PSR_psr_spike': 0.058823529,
        'Stability_gain': 0.117647059,
        'Stability_marginGain': 0.058823529,
        'Stability_phaseMargin': 0.058823529,
        'Stability_gainBandWidth': 0.117647059,
        'Output_Tolerance_1u_outputTolerance': 0.058823529,
        'Output_Tolerance_20m_outputTolerance': 0.058823529,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_DRMOS_Legacy(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'Efficiency_16A_inputPower': 0.02,
        'Efficiency_16A_outputPower': 0.02,
        'Efficiency_16A_efficiency': 0.16,
        'Efficiency_18A_inputPower': 0.02,
        'Efficiency_18A_outputPower': 0.02,
        'Efficiency_18A_efficiency': 0.16,
        'Efficiency_20A_inputPower': 0.02,
        'Efficiency_20A_outputPower': 0.02,
        'Efficiency_20A_efficiency': 0.16,
        'Efficiency_22A_inputPower': 0.02,
        'Efficiency_22A_outputPower': 0.02,
        'Efficiency_22A_efficiency': 0.16,
        'Trans_deadTime': 0.025,
        'Trans_riseTime': 0.025,
        'Trans_averageVoltage': 0.05,
        'Area_area': 0.1
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_LFM(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.0097,
        'Stability_100u_gainBandWidth': 0.0208,
        'Stability_100u_phaseMargin': 0.0995,
        'Stability_1m_gainBandWidth': 0.0208,
        'Stability_1m_phaseMargin': 0.0995,
        'Stability_10m_gainBandWidth': 0.0208,
        'Stability_10m_phaseMargin': 0.0995,
        'Stability_100m_gainBandWidth': 0.0208,
        'Stability_100m_phaseMargin': 0.0995,
        'Stability_300m_gainBandWidth': 0.0208,
        'Stability_300m_phaseMargin': 0.0995,
        'Trans_overShoot': 0.0097,
        'Trans_underShoot': 0.0097,
        'PSR_100u_psr_1M': 0.0462,
        'PSR_100u_psr_10M': 0.0462,
        'PSR_100u_psr_100M': 0.0462,
        'PSR_100u_psr_1G': 0.0462,
        'PSR_300m_psr_1M': 0.0462,
        'PSR_300m_psr_10M': 0.0462,
        'PSR_300m_psr_100M': 0.0462,
        'PSR_300m_psr_1G': 0.0462,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_DrMOS(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'Efficiency_0.5A_efficiency': 0.1,
        'Efficiency_0.5A_vo_mean': 0.1,
        'Efficiency_0.5A_vo_variation_percent': 0.1,
        'Efficiency_1A_efficiency': 0.1,
        'Efficiency_1A_vo_mean': 0.1,
        'Efficiency_1A_vo_variation_percent': 0.1,
        'Efficiency_2A_efficiency': 0.1,
        'Efficiency_2A_vo_mean': 0.1,
        'Efficiency_2A_vo_variation_percent': 0.1,
        'Area_area': 0.1
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict, reward_weight,
                          reward_mode="standard",
                          constraint_penalty_range=5,
                          optimization_bonus_range=10,
                          invalid_penalty_range=5):
    """
    Generalized reward calculation function that supports configurable reward weights and ranges.
    Support three types of objectives:
    - max: maximize the value
    - min: minimize the value
    - range: keep value within specified range

    Args:
        ideal_specs_dict: Dict with ideal specs value and property
        cur_specs_dict: Dict with current specs
        norm_specs_dict: Dict with normalized specs
        reward_weight: Dict with reward weights for each spec
        reward_mode: str, "standard" or "invalid_penalty" (default: "standard")
        constraint_penalty_range: Range multiplier for negative rewards (default: 5)
        optimization_bonus_range: Range multiplier for positive bonus rewards (default: 10)
        invalid_penalty_range: Range multiplier for invalid value penalties (default: 5)
    Returns:
        reward: float, reward value
    """
    logging.debug("Calculating reward using generalized function")
    logging.debug(f"Ideal specs: {ideal_specs_dict}")
    logging.debug(f"Current specs: {cur_specs_dict}")

    # Fixed invalid values list
    INVALID_VALUES = [0, 100]

    # Flatten cur_specs_dict
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}
    norm_specs_flatten = {k: v for d in norm_specs_dict.values() for k, v in d.items()}

    # Check for invalid values if using invalid_penalty mode
    if reward_mode == "invalid_penalty":
        invalid_specs = []
        for spec, value in cur_specs_flatten.items():
            if value in INVALID_VALUES:
                invalid_specs.append(spec)

        if invalid_specs:
            invalid_count = len(invalid_specs)
            all_count = len(cur_specs_flatten)
            logging.info(f"Found {invalid_count} invalid values in specs: {invalid_specs}")

            # First stage: invalid value penalty
            rew = (-invalid_penalty_range - constraint_penalty_range +
                   (all_count - invalid_count) / all_count * invalid_penalty_range)
            return rew

    rew = 0

    for spec, detail in ideal_specs_dict.items():
        single_reward = 0
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_objective = detail['objective']

        if constrain_objective == "range":
            # Get range bounds from value list
            range_min, range_max = detail['value']
            if range_min <= cur_spec_value <= range_max:
                single_reward = 0.0
            else:
                # If below minimum, treat as max objective
                if cur_spec_value < range_min:
                    single_reward = min((cur_spec_value - range_min) / (cur_spec_value + range_min), 0.0)
                # If above maximum, treat as min objective
                else:
                    single_reward = min((range_max - cur_spec_value) / (cur_spec_value + range_max), 0.0)
        else:
            # Handle max/min objectives
            ideal_spec_value = float(detail['value'])
            if constrain_objective == "max":
                single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
            elif constrain_objective == "min":
                single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)

        weighted_single_reward = single_reward * reward_weight[spec]
        rew += float(weighted_single_reward)

    rew = rew * constraint_penalty_range

    if rew >= 0:
        rew_base = 10
        rew_bonus = 0
        for spec, detail in ideal_specs_dict.items():
            single_reward = 0
            reward_type = detail['reward_type']

            # Only calculate bonus reward for optimal type and non-range objectives
            if reward_type == "optimal" and detail['objective'] != "range":
                general_ideal_spec_value = float(norm_specs_flatten[spec])
                cur_spec_value = float(cur_specs_flatten[spec])
                constrain_objective = detail['objective']

                if constrain_objective == "max":
                    single_reward = max(
                        (cur_spec_value - general_ideal_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
                elif constrain_objective == "min":
                    single_reward = max(
                        (general_ideal_spec_value - cur_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)

                weighted_single_reward = single_reward * reward_weight[spec]
                rew_bonus += float(weighted_single_reward)

        rew_bonus = rew_bonus * optimization_bonus_range
        rew = rew_base + rew_bonus

    return rew

def cal_reward_AMP_LG(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.096,
        'PSR_psr_10': 0.016,
        'PSR_psr_1k': 0.016,
        'PSR_psr_1M': 0.016,
        'Stability_gain': 0.231,
        'Stability_phaseMargin': 0.044,
        'Stability_bandwidth': 0.581,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_CMP_LG(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.2,
        'Trans_rise_delay_time': 0.4,
        'Trans_fall_delay_time': 0.4,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_CMP_hysteretic(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.058823529,
        'Trans_rise_delay_time': 0.292527822,
        'Trans_fall_delay_time': 0.292527822,
        'Hysteretic_hysteretic_high_voltage': 0.138844727,
        'Hysteretic_hysteretic_low_voltage': 0.138844727,
        'PSR_psr_1k': 0.039215686,
        'PSR_psr_1M': 0.039215686,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_CMP_hysteretic_Simple(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.081395349,
        'Trans_rise_delay_time': 0.404825991,
        'Trans_fall_delay_time': 0.404825991,
        'PSR_psr_1k': 0.054476335,
        'PSR_psr_1M': 0.054476335,
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)

def cal_reward_AMP_Three_Stage_Block(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    Simplified version using cal_reward_ahp_general.
    """
    reward_weight = {
        'DC_IQ': 0.262,
        'Stability_unityGainFreq': 0.223,
        'Stability_phaseMargin': 0.041,
        'Stability_gainMargin': 0.041,
        'Stability_dcGain': 0.172,
        'Trans_slewRate': 0.145,
        'Trans_onePercentSettlingTime': 0.113
    }
    return cal_reward_ahp_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict,
                                 reward_weight, reward_mode="standard",
                                 constraint_penalty_range=5, optimization_bonus_range=10)
# if __name__ == '__main__':
#     ideal_specs_file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/ideal_specs/sampled_specs_DRMOS/1.yaml"
#     with open(ideal_specs_file, 'r') as f:
#         ideal_specs = yaml.load(f, Loader=yaml.FullLoader)
#     cur_specs = {'Efficiency_16A': {'Efficiency_16A_inputPower': 30, 'Efficiency_16A_outputPower': 30, 'Efficiency_16A_efficiency': 0.98}, 'Efficiency_18A': {'Efficiency_18A_inputPower': 30, 'Efficiency_18A_outputPower': 30, 'Efficiency_18A_efficiency': 0.98}, 'Efficiency_20A': {'Efficiency_20A_inputPower': 30, 'Efficiency_20A_outputPower': 30, 'Efficiency_20A_efficiency': 0.98}, 'Efficiency_22A': {'Efficiency_22A_inputPower': 30, 'Efficiency_22A_outputPower': 30, 'Efficiency_22A_efficiency': 0.98}, 'Trans': {'Trans_deadTime': 1.01e-09, 'Trans_riseTime': 4.4e-10, 'Trans_averageVoltage': 0.904}, 'Area': {'Area_area': 2.0531325000000002}}
#     norm_specs_file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_DRMOS/norm_specs.yaml"
#     with open(norm_specs_file, 'r') as f:
#         norm_specs = yaml.load(f, Loader=yaml.FullLoader)
#     reward = cal_reward_DRMOS(ideal_specs, cur_specs, norm_specs)
#     print(reward)



def cal_reward_BGR(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """Reward for BGR curve-comp topology with explicit priority on TC and Vref."""
    reward_weight = {}
    for spec in ideal_specs_dict:
        if spec == "DC_TC":
            reward_weight[spec] = 6.0
        elif spec == "DC_Vref":
            reward_weight[spec] = 6.0
        elif spec == "DC_IQ":
            reward_weight[spec] = 2.0
        else:
            reward_weight[spec] = 1.0

    return cal_reward_ahp_general(
        ideal_specs_dict,
        cur_specs_dict,
        norm_specs_dict,
        reward_weight,
        reward_mode="standard",
        constraint_penalty_range=5,
        optimization_bonus_range=10,
    )


def cal_reward_BGR(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """BGR reward with normalized negative range in [-5, 0] and positive bonus above 10."""
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}
    norm_specs_flatten = {k: v for d in norm_specs_dict.values() for k, v in d.items()}

    reward_weight = {}
    for spec in ideal_specs_dict:
        if spec == "DC_TC":
            reward_weight[spec] = 6.0
        elif spec == "DC_Vref":
            reward_weight[spec] = 6.0
        elif spec == "DC_IQ":
            reward_weight[spec] = 2.0
        else:
            reward_weight[spec] = 1.0

    weight_sum = max(sum(reward_weight.values()), 1e-9)

    rew_raw = 0.0
    for spec, detail in ideal_specs_dict.items():
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_objective = detail[objective]
        single_reward = 0.0

        if constrain_objective == "range":
            range_min, range_max = detail[value]
            if not (range_min <= cur_spec_value <= range_max):
                if cur_spec_value < range_min:
                    single_reward = min((cur_spec_value - range_min) / (cur_spec_value + range_min), 0.0)
                else:
                    single_reward = min((range_max - cur_spec_value) / (cur_spec_value + range_max), 0.0)
        else:
            ideal_spec_value = float(detail[value])
            if constrain_objective == "max":
                single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
            elif constrain_objective == "min":
                single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)

        rew_raw += single_reward * reward_weight[spec]

    # Normalized negative range: [-5, 0]
    rew = (rew_raw / weight_sum) * 5.0

    if rew >= 0:
        rew = 10.0
        rew_bonus = 0.0
        for spec, detail in ideal_specs_dict.items():
            if detail[reward_type] != optimal or detail[objective] == range:
                continue

            general_ideal_spec_value = float(norm_specs_flatten[spec])
            cur_spec_value = float(cur_specs_flatten[spec])
            constrain_objective = detail[objective]
            single_reward = 0.0

            if constrain_objective == "max":
                single_reward = max((cur_spec_value - general_ideal_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
            elif constrain_objective == "min":
                single_reward = max((general_ideal_spec_value - cur_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)

            rew_bonus += single_reward * reward_weight[spec]

        rew += (rew_bonus / weight_sum) * 10.0

    return rew


def cal_reward_BGR(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
    """BGR reward normalized to negative range [-5, 0], with positive bonus above 10."""
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}
    norm_specs_flatten = {k: v for d in norm_specs_dict.values() for k, v in d.items()}

    reward_weight = {}
    for spec in ideal_specs_dict:
        if spec == 'DC_TC':
            reward_weight[spec] = 6.0
        elif spec == 'DC_Vref':
            reward_weight[spec] = 6.0
        elif spec == 'DC_IQ':
            reward_weight[spec] = 2.0
        else:
            reward_weight[spec] = 1.0

    weight_sum = max(sum(reward_weight.values()), 1e-9)

    rew_raw = 0.0
    for spec, detail in ideal_specs_dict.items():
        cur_spec_value = float(cur_specs_flatten[spec])
        constrain_objective = detail['objective']
        single_reward = 0.0

        if constrain_objective == 'range':
            range_min, range_max = detail['value']
            if not (range_min <= cur_spec_value <= range_max):
                if cur_spec_value < range_min:
                    single_reward = min((cur_spec_value - range_min) / (cur_spec_value + range_min), 0.0)
                else:
                    single_reward = min((range_max - cur_spec_value) / (cur_spec_value + range_max), 0.0)
        else:
            ideal_spec_value = float(detail['value'])
            if constrain_objective == 'max':
                single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)
            elif constrain_objective == 'min':
                single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0.0)

        rew_raw += single_reward * reward_weight[spec]

    rew = (rew_raw / weight_sum) * 5.0

    if rew >= 0:
        rew = 10.0
        rew_bonus = 0.0
        for spec, detail in ideal_specs_dict.items():
            if detail['reward_type'] != 'optimal' or detail['objective'] == 'range':
                continue

            general_ideal_spec_value = float(norm_specs_flatten[spec])
            cur_spec_value = float(cur_specs_flatten[spec])
            constrain_objective = detail['objective']
            single_reward = 0.0

            if constrain_objective == 'max':
                single_reward = max((cur_spec_value - general_ideal_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)
            elif constrain_objective == 'min':
                single_reward = max((general_ideal_spec_value - cur_spec_value) / (cur_spec_value + general_ideal_spec_value), 0.0)

            rew_bonus += single_reward * reward_weight[spec]

        rew += (rew_bonus / weight_sum) * 10.0

    return rew
