def cal_reward_buck(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
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
    max_vo = 1
    reward_weight = {}
    for spec, detail in ideal_specs_dict.items():
        reward_weight[spec] = 1
    reward_weight_sum = sum(reward_weight.values())

    rew = 0
    if cur_specs_flatten['Efficiency_vo_mean'] > max_vo:
        rew = min_rew_bound
        print("Warning! The buck is not regulated normally")
    else:
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
#                'Efficiency_vo_var': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.0000011}}
#
# cur_specs = {'Efficiency': {'Efficiency_efficiency': 0.95, 'Efficiency_vo_mean': 0.94, 'Efficiency_vo_var': 0.000001}}
#
# norm_specs = {
#     'Efficiency': {'Efficiency_efficiency': 0.92, 'Efficiency_vo_mean': 0.927, 'Efficiency_vo_var': 0.0000011}}
#
# reward = cal_reward_buck(ideal_specs, cur_specs, norm_specs)
# print(reward)

def cal_reward_general(ideal_specs_dict, cur_specs_dict, norm_specs_dict):
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
#                'Efficiency_vo_var': {'objective': 'min', 'reward_type': 'optimal', 'value': 0.0000011}}
#
# cur_specs = {'Efficiency': {'Efficiency_efficiency': 0.95, 'Efficiency_vo_mean': 1.1, 'Efficiency_vo_var': 0.000001}}
#
# norm_specs = {
#     'Efficiency': {'Efficiency_efficiency': 0.92, 'Efficiency_vo_mean': 0.927, 'Efficiency_vo_var': 0.0000011}}
#
# reward = cal_reward_general(ideal_specs, cur_specs, norm_specs)
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
# cur_specs = {'Efficiency': {'Efficiency_efficiency': 0.95, 'Efficiency_vo_mean': 0.93, 'Efficiency_vo_var': 0.000001},
#              'Load_Reg': {'Load_Reg_loadReg': 1.1}, 'Trans': {'Trans_overShoot': 1.1}, 'DC': {'DC_IQ': 1.1}}
#
# norm_specs = {'Efficiency': {'Efficiency_efficiency': 0.93, 'Efficiency_vo_mean': 1.1, 'Efficiency_vo_var': 0.000001},
#               'Load_Reg': {'Load_Reg_loadReg': 1}, 'Trans': {'Trans_overShoot': 1}, 'DC': {'DC_IQ': 1}}
#
# reward = cal_reward_LDO(ideal_specs, cur_specs, norm_specs)
# print(reward)
