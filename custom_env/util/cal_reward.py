import yaml


def cal_reward(ideal_specs_dict, cur_specs_dict):
    """
    Calculate the reward based on the ideal specs and current specs.
    :param ideal_specs_dict: Dict with ideal specs value and property
    :param cur_specs_dict: Dict with current specs
    :return: reward: float, reward value
    """

    # Flatten cur_specs_dict
    cur_specs_flatten = {k: v for d in cur_specs_dict.values() for k, v in d.items()}

    reward = 0

    epsilon = 0.01

    for spec, detail in ideal_specs_dict.items():

        single_reward = 0

        ideal_spec_value = detail['value']
        cur_spec_value = cur_specs_flatten[spec]
        constrain_type = detail['constrain_type']
        constrain_objective = detail['objective']

        if constrain_type == 'hard' and constrain_objective == "max":
            single_reward = min((cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value), 0)
        elif constrain_type == 'hard' and constrain_objective == "min":
            single_reward = min((ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value), 0)
        elif constrain_type == 'soft' and constrain_objective == "max":
            single_reward = epsilon * (ideal_spec_value - cur_spec_value) / (cur_spec_value + ideal_spec_value)
        elif constrain_type == 'soft' and constrain_objective == "min":
            single_reward = epsilon * (cur_spec_value - ideal_spec_value) / (cur_spec_value + ideal_spec_value)

        reward += single_reward

    reward = reward + 10 if reward >= -0.01 else reward
    reward = float(reward)

    return reward


# Test Code
# ideal_specs = {'gainBandWidth': {'constrain_type': 'hard', 'objective': 'max', 'value': 1},
# 'phaseMargin': {'constrain_type': 'hard', 'objective': 'max', 'value': 2}, 'powerSupplyRejectionRatio': {
# 'constrain_type': 'hard', 'objective': 'max', 'value': 3}, 'pwr': {'constrain_type': 'soft', 'objective': 'min',
# 'value': 4}, 'slewRateDown': {'constrain_type': 'hard', 'objective': 'max', 'value': 5}, 'slewRateUp': {
# 'constrain_type': 'hard', 'objective': 'max', 'value': 6}}
# cur_specs = {'DC': {'pwr': 1}, 'Stability': {
# 'phaseMargin': 2, 'gainBandWidth': 3}, 'Trans': {'slewRateUp': 4, 'slewRateDown': 5}, 'PSRR': {
# 'powerSupplyRejectionRatio': 6}} reward = cal_reward(ideal_specs, cur_specs) print(reward)

# Output
# -0.20600000000000002
