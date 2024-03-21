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
ideal_specs = {'gainBandWidth': {'reward_type': 'optimal', 'objective': 'max', 'value': 1},
               'phaseMargin': {'reward_type': 'satisfactory', 'objective': 'max', 'value': 2},
               'powerSupplyRejectionRatio': {'reward_type': 'optimal', 'objective': 'max', 'value': 3},
               'pwr': {'reward_type': 'optimal', 'objective': 'min','value': 4}}
cur_specs = {'DC': {'pwr': 1}, 'Stability': {'phaseMargin': 2, 'gainBandWidth': 3},
              'PSRR': {'powerSupplyRejectionRatio': 4}}
norm_specs = {'DC': {'pwr': 2}, 'Stability': {'phaseMargin': 2, 'gainBandWidth': 3},
              'PSRR': {'powerSupplyRejectionRatio': 4}}
reward = cal_reward(ideal_specs, cur_specs, norm_specs)
print(reward)

# Output
# -0.20600000000000002

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